"""Instance-local growth policy; never patch or write the original bundle.

Compile a private copy with checked growth and reporting changes.
Candidate search, gating and fitting remain native.
"""
import ast
import functools
import inspect
import textwrap


def grow_until_neurons(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.While)]
    failures = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and ast.unparse(node.test) == 'flag == 1']
    checkpoints = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "checkpoint_metrics['accuracy'] > best_checkpoint_accuracy"]
    if len(loops) != 1 or len(failures) != 1 or len(checkpoints) != 1:
        raise RuntimeError('Native GateZO control flow changed; review growth policy adapter')
    loop, failure, checkpoint = loops[0], failures[0], checkpoints[0]
    expected = 'model.L < model.L_max and error > model.tol and (current_window_best - previous_window_best >= 0.001)'
    if ast.dump(loop.test) != ast.dump(ast.parse(expected, mode='eval').body):
        raise RuntimeError('Native GateZO stopping conditions changed; review adapter')
    if not isinstance(failure.body[-1], ast.Break):
        raise RuntimeError('Native GateZO search failure handling changed; review adapter')
    reporting = [node for node in loop.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'runtime'
                 and any(isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
                         and ast.unparse(child.value.func) == 'runtime.epoch_row'
                         for child in node.body)]
    if len(reporting) != 1:
        raise RuntimeError('Native GateZO reporting block changed; review adapter')
    block = reporting[0]
    expected_reporting = ast.parse('''
runtime.end(train_metrics['accuracy'], train_metrics['mse'])
runtime.begin('test', attempts)
test_h, _ = model.getH(X_test)
observed = classification_metrics(np.asarray(test_h), T_test, model.Beta, _counter(model))
runtime.end(observed['accuracy'], observed['mse'])
runtime.epoch_row(attempts, unit='growth_round')
runtime.rows[-1]['validation_accuracy'] = checkpoint_metrics['accuracy']
runtime._csv()
''').body
    if [ast.dump(n) for n in block.body] != [ast.dump(n) for n in expected_reporting]:
        raise RuntimeError('Native GateZO test reporting changed; review adapter')
    # Keep the train-side metrics required by structural selection. Remove only
    # held-out test reporting, including its forward pass, from every growth round.
    block.body = [block.body[0], *block.body[5:]]
    loop.test = ast.parse('model.L < model.L_max and model.hidden_count < model.max_hidden',
                          mode='eval').body
    # Preserve the native break on candidate-search failure. No unlimited retries.
    # Train/validation computations used by structure selection stay inside the
    # continuous training window. No per-round held-out test is executed.
    # Keep the target-sized checkpoint, even when a smaller checkpoint scored better.
    # Otherwise native final restoration would undo the requested growth target.
    checkpoint.test = ast.BoolOp(op=ast.And(), values=[
        ast.parse('model.hidden_count == model.max_hidden', mode='eval').body,
        checkpoint.test])
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, '<texture_experiments.growth_policy>', 'exec'), namespace)
    adapted = namespace[function.__name__]
    native_refit = namespace['refit_and_measure']

    def refit_excluding_test_measurement(model, x, targets, x_test, test_targets, ridge):
        runtime = getattr(model, '_pp3_runtime', None)
        if runtime is None:
            return native_refit(model, x, targets, x_test, test_targets, ridge)
        train_h, _ = model.getH(x)
        beta = namespace['fit_readout'](
            namespace['np'].asarray(train_h), targets, ridge, namespace['_counter'](model)
        )
        model.Beta = beta
        runtime.finish_measurement()
        train_metrics = namespace['classification_metrics'](
            namespace['np'].asarray(train_h), targets, beta, namespace['_counter'](model)
        )
        test_h, _ = model.getH(x_test)
        test_metrics = namespace['classification_metrics'](
            namespace['np'].asarray(test_h), test_targets, beta, namespace['_counter'](model)
        )
        runtime.final_evaluation = (train_metrics, test_metrics)
        return train_metrics, test_metrics

    # The adapted growth function resolves this name in its private namespace.
    namespace['refit_and_measure'] = refit_excluding_test_measurement

    @functools.wraps(function)
    def wrapped(model, *args, **kwargs):
        if model.structural_gate.config.growth_protocol != 'validation_checkpoint':
            raise ValueError('Texture energy experiments require validation_checkpoint')
        if model.max_hidden < 1 or model.hidden_count >= model.max_hidden:
            raise ValueError('Growth requires a positive target above the initial hidden count')
        result = adapted(model, *args, **kwargs)
        if model.hidden_count != model.max_hidden:
            raise RuntimeError('GateZO stopped before its neuron target: '
                               f'L={model.L}, L_max={model.L_max}, '
                               f'hidden={model.hidden_count}, target={model.max_hidden}')
        return result
    return wrapped

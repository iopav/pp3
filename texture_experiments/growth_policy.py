"""Instance-local growth policy; never patch or write the original bundle.

Compile a private copy of the native function with three checked control-flow
changes. Candidate search, gating, fitting and accounting remain native.
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
    metric_positions = [index for index, node in enumerate(loop.body)
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'train_metrics'
                                for target in node.targets)]
    if len(metric_positions) != 1:
        raise RuntimeError('Native GateZO train-metric position changed; review adapter')
    loop.test = ast.parse('model.hidden_count < model.max_hidden', mode='eval').body
    failure.body[-1] = ast.Continue()
    # Candidate search, structural selection and readout fitting are measured;
    # reporting evaluations on training/validation/test data are not.
    stop_measurement = ast.parse(
        'if runtime:\n    runtime.finish_measurement()'
    ).body[0]
    loop.body.insert(metric_positions[0], stop_measurement)
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
        return train_metrics, test_metrics

    # The adapted growth function resolves this name in its private namespace.
    namespace['refit_and_measure'] = refit_excluding_test_measurement

    @functools.wraps(function)
    def wrapped(model, *args, **kwargs):
        if model.max_hidden < 1 or model.hidden_count >= model.max_hidden:
            raise ValueError('Growth requires a positive target above the initial hidden count')
        result = adapted(model, *args, **kwargs)
        if model.hidden_count != model.max_hidden:
            raise RuntimeError('GateZO returned without reaching its neuron target')
        return result
    return wrapped

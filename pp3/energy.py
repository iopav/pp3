"""Device energy only. Never attribute CPU energy to the GPU."""
import threading
import time
import torch


class Energy:
    def __init__(self,device,interval=.1):
        self.device=device
        self.interval=interval
        self.reason='CPU run: GPU energy unavailable'
        self.mode='unavailable'
        self.samples=[]
        self.stop_event=threading.Event()
        self.thread=None
        self.nv=None
        if device.type!='cuda':
            return
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv=nv
            props=torch.cuda.get_device_properties(device)
            uuid=str(props.uuid)
            if not uuid.startswith(('GPU-','MIG-')):
                uuid='GPU-'+uuid
            self.handle=nv.nvmlDeviceGetHandleByUUID(uuid)
            self.uuid=uuid
            try:
                nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.mode='nvml_total'
            except nv.NVMLError:
                nv.nvmlDeviceGetPowerUsage(self.handle)
                self.mode='power_integral_estimate'
            self.reason=''
        except Exception as e:
            self.reason=f'{type(e).__name__}: {e}'

    def synchronize(self):
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)

    def sample(self):
        try:
            self.samples.append((time.perf_counter(),self.nv.nvmlDeviceGetPowerUsage(self.handle)/1000))
        except Exception as e:
            self.reason=str(e)

    def poll(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.synchronize()
        self.start=time.perf_counter()
        try:
            if self.mode=='nvml_total':
                self.initial=self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            elif self.mode=='power_integral_estimate':
                self.sample()
                self.thread=threading.Thread(target=self.poll,daemon=True)
                self.thread.start()
        except Exception as e:
            self.reason=str(e)
            self.mode='unavailable'
        return self

    def __exit__(self,*exc):
        self.synchronize()
        self.seconds=time.perf_counter()-self.start
        self.joules=None
        try:
            if self.mode=='nvml_total':
                delta=self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)-self.initial
                if delta<0:
                    raise ValueError('GPU energy counter reset')
                self.joules=delta/1000
                if delta==0:
                    self.reason='Counter did not advance; interval may be shorter than hardware update period'
            elif self.mode=='power_integral_estimate':
                self.stop_event.set()
                self.thread.join()
                self.sample()
                if not self.reason:
                    self.joules=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(self.samples,self.samples[1:]))
        except Exception as e:
            self.reason=str(e)
        finally:
            if self.nv:
                self.nv.nvmlShutdown()

    def result(self):
        return dict(seconds=self.seconds,gpu_energy_j=self.joules,energy_method=self.mode,
                    energy_reason=self.reason,gpu_uuid=getattr(self,'uuid',None))

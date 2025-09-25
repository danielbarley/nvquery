import pynvml
from pynvml import _nvmlSamplingType_t
from time import sleep


_SAMPLE_VALUE_TYPE_TO_FIELD = {
    0: "dVal",
    1: "uiVal",
    2: "ulVal",
    3: "ullVal",
    4: "sllVal",
    5: "siVal",
    6: "usVal",
}

_SAMPLING_TYPE_TO_HEADER = {
    pynvml.NVML_TOTAL_POWER_SAMPLES: "Power",
    pynvml.NVML_MEMORY_UTILIZATION_SAMPLES: "Memory Util",
    pynvml.NVML_GPU_UTILIZATION_SAMPLES: "GPU Util",
    pynvml.NVML_PROCESSOR_CLK_SAMPLES: "GPU Clk",
    pynvml.NVML_MEMORY_CLK_SAMPLES: "Memory Clk",
}


class NvmlReader:
    def __init__(
        self,
        log_file_base_name: str = "log",
        sampling_type: _nvmlSamplingType_t = pynvml.NVML_TOTAL_POWER_SAMPLES,
    ):
        if sampling_type not in [
            pynvml.NVML_TOTAL_POWER_SAMPLES,
            pynvml.NVML_MEMORY_UTILIZATION_SAMPLES,
            pynvml.NVML_GPU_UTILIZATION_SAMPLES,
            pynvml.NVML_PROCESSOR_CLK_SAMPLES,
            pynvml.NVML_MEMORY_CLK_SAMPLES,
        ]:
            raise ValueError("Illegal or unsupported sampling type")
        self.sampling_type = sampling_type
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = []
        self.logs = []
        self.last_sample_time = 0
        for device_id in range(self.device_count):
            self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(device_id))
        if len(self.handles) != self.device_count:
            raise RuntimeError("Failed to get all device handles")
        for idx, _ in enumerate(self.handles):
            filename = log_file_base_name + "_" + str(idx) + ".csv"
            self.logs.append(open(filename, "w"))
        # Perform test read on first device to get sampling_type
        sample_value_type, _ = pynvml.nvmlDeviceGetSamples(
            device=self.handles[0],
            sampling_type=self.sampling_type,
            timeStamp=0,
        )
        self.field = _SAMPLE_VALUE_TYPE_TO_FIELD.get(sample_value_type)
        self.metric = _SAMPLING_TYPE_TO_HEADER.get(sampling_type)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for log in self.logs:
            log.close()

    def print_clocks(self):
        for id, handle in enumerate(self.handles):
            print(f" Device: {id} ".center(80, "="))
            print("Memory Clock - Possible Compute Clocks")
            supported_memory_clocks = (
                pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
            )
            for memory_clock in supported_memory_clocks:
                supported_compute_clocks = (
                    pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                        handle, memory_clock
                    )
                )
                print("-" * 80)
                print(
                    f"{str(memory_clock).ljust(12)} - {supported_compute_clocks}"
                )
                print(f"{pynvml.nvmlDeviceGetCurrentClockFreqs(handle)}")

    def print_arch(self):
        for id, handle in enumerate(self.handles):
            print(
                f"Device {id}: Arch: {pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, 1215)}"
            )

    def print_current_clock(self):
        for id, handle in enumerate(self.handles):
            print(f" Device: {id} ".center(80, "="))
            for clock_type, name in {
                pynvml.NVML_CLOCK_SM: "SM Clock",
                pynvml.NVML_CLOCK_MEM: "Memory Clock",
            }.items():
                print(
                    name,
                    pynvml.nvmlDeviceGetClock(
                        handle,
                        type=clock_type,
                        id=pynvml.NVML_CLOCK_ID_CURRENT,
                    ),
                )

    def print_current_utilization(self):
        for id, handle in enumerate(self.handles):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"Device {id}: Util: gpu {util.gpu}%, mem {util.memory}%")

    def print_current_power(self):
        for id, handle in enumerate(self.handles):
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            print(f"Device {id}: Power: {power / 1e3}W")

    def print_current_samples(self):
        for id, handle in enumerate(self.handles):
            _, samples = pynvml.nvmlDeviceGetSamples(
                device=handle,
                sampling_type=self.sampling_type,
                timeStamp=self.last_mem_sample_time,
            )
            print(
                f"{len(samples)} samples "
                f"over {(samples[-1].timeStamp - samples[0].timeStamp) / 1e3}ms\n"
            )

    def log_header(self):
        header_string = f"Time,{self.metric}\n"
        for log in self.logs:
            log.write(header_string)

    def log_samples(self):
        for id, handle in enumerate(self.handles):
            try:
                _, samples = pynvml.nvmlDeviceGetSamples(
                    device=handle,
                    sampling_type=self.sampling_type,
                    timeStamp=self.last_sample_time,
                )
                self.last_sample_time = samples[-1].timeStamp
                for sample in samples:
                    self.logs[id].write(
                        f"{sample.timeStamp},{getattr(sample.sampleValue, self.field)}\n"
                    )
            except pynvml.NVMLError as e:
                print(f"WARNING nvml error during sampling: {e}")

    def set_last_seen(self):
        for id, handle in enumerate(self.handles):
            try:
                _, samples = pynvml.nvmlDeviceGetSamples(
                    device=handle,
                    sampling_type=self.sampling_type,
                    timeStamp=self.last_sample_time,
                )
                self.last_sample_time = samples[-1].timeStamp
            except pynvml.NVMLError as e:
                print(f"WARNING nvml error: {e}")


if __name__ == "__main__":
    power_reader = NvmlReader("power")
    power_reader.print_clocks()

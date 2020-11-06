import csv

import numpy
from hrv.rri import RRi
from hrv.utils import _create_interp_time, _interp_cubic_spline
from biosppy import ecg
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation


ECG_FILE_NAME = "Patient05_reduced_data.txt"
FS = 512
SHOW_ECG_TRACE = False
RRI_RESAMPLING_FS = 4

SEG_SIZE = 180
OVERLAP = 90


print("Reading file...")
with open(ECG_FILE_NAME) as fobj:
    # skip header
    fobj.readline()
    reader = csv.reader(fobj)
    raw_ecg_signal = [float(row[1]) for row in reader]


print("Detecting R-peaks...")
extraction = ecg.ecg(raw_ecg_signal, sampling_rate=512, show=SHOW_ECG_TRACE)
print("Done!")
rri_series = numpy.diff(extraction["rpeaks"]) * (1 / FS) * 1000
time_info = numpy.cumsum(rri_series) / 1000.0
time_info -= time_info[0]

rrix = _interp_cubic_spline(rri_series, time_info, RRI_RESAMPLING_FS)
time_interp = _create_interp_time(time_info, RRI_RESAMPLING_FS)
rri_obj = RRi(rrix, time=time_interp)

print("Processing RQA")
for idx, segment in enumerate(rri_obj.split_time(SEG_SIZE, OVERLAP)):
    time_series = TimeSeries(
        segment.values,
        embedding_dimension=2,
        time_delay=2
    )

    settings = Settings(
        time_series,
        analysis_type=Classic,
        neighbourhood=FixedRadius(0.65),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1
    )
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    print(result)

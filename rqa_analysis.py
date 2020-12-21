import csv

import numpy
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from scipy import interpolate


RRI_PATH_NAME = ""
RRI_RESAMPLING_FS = 4

SEG_SIZE = 180
OVERLAP = 90


def create_time_info(rri):
    rri_time = numpy.cumsum(rri) / 1000.0  # make it seconds
    return rri_time - rri_time[0]  # force it to start at zero


def create_interp_time(time, fs):
    time_resolution = 1 / float(fs)
    return numpy.arange(0, time[-1] + time_resolution, time_resolution)


def interp_cubic_spline(rri, time, fs):
    time_rri_interp = create_interp_time(time, fs)
    tck = interpolate.splrep(time, rri, s=0)
    rri_interp = interpolate.splev(time_rri_interp, tck, der=0)
    return time_rri_interp, rri_interp


def time_split(rri, time, seg_size, overlap=0, keep_last=False):
    rri_duration = time[-1]

    begin = 0
    end = seg_size
    step = seg_size - overlap
    n_splits = int((rri_duration - seg_size) / step) + 1
    rri_segments = []
    time_segments = []
    for i in range(n_splits):
        OP = numpy.less if i + 1 != n_splits else numpy.less_equal
        mask = numpy.logical_and(time >= begin, OP(time, end))
        rri_segments.append(rri[mask])
        time_segments.append(time[mask])
        begin += step
        end += step

    last = time_segments[-1][-1]
    if keep_last and last < rri_duration:
        mask = time > begin
        rri_segments.append(rri[mask])
        time_segments.append(time[mask])

    return time_segments, rri_segments


rri = numpy.loadtxt(RRI_PATH_NAME)
time = create_time_info(rri)

time_x, rri_x = interp_cubic_spline(rri, time, RRI_RESAMPLING_FS)

time_s, rri_s = time_split(rri_x, time_x, SEG_SIZE, OVERLAP, keep_last=True)

print("Processing RQA")
for i, segment in enumerate(rri_s):
    print("-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")
    print(f"Analyzing segment #{i}")
    time_series = TimeSeries(
        segment,
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

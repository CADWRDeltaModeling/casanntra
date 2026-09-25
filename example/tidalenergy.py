from dms_datastore.read_multi import read_ts_repo
from vtools import *
from vtools.functions.unit_conversions import M2FT

mrz = read_ts_repo("mrz", "elev", "upper")
sf = read_ts_repo("sffpx", "elev")
print(f"SF mean {sf.mean()}, SF st. dev {sf.std()}")
print(f"MRZ mean {mrz.mean()}, MRZ st. dev {mrz.std()}")


def tidal_energy(ts):
    mean = cosine_lanczos(ts, "40h")
    ts0 = (ts - mean) * (ts - mean)
    return cosine_lanczos(ts, "40h")


print(mrz.mean(axis=None))
print(sf.mean(axis=None))

print(tidal_energy(mrz).mean(axis=None))
print(tidal_energy(sf).mean(axis=None))

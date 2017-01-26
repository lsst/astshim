# generate includes and data for CastMap in Object.cc

import glob
import os.path

root_dir = os.path.dirname(os.path.dirname(__file__))
inc_dir = os.path.join(root_dir, "include", "astshim")
path_list = glob.glob(inc_dir + "/*.h")
name_list = [os.path.splitext(os.path.basename(path))[0] for path in path_list]
name_list.sort()

ExtraIncludes = [  # without the ".h" suffix
    "base",
    "detail/utils",
    "Object",
]

# files to skip (minus trailing ".h")
SkipNames = set([
    # Code does not wrap AST classes
    "base",
    "MapBox",
    "MapSplit",
    "QuadApprox",
    "Stream",

    # Excluded objects
    "CmpMap",  # always returns as SeriesMap or ParallelMap
    "KeyMap",  # unfinished
    "Mapping",  # abstract base class
    "Object",  # abstract base class

    # Channels cannot be persisted because streams cannot
    "Channel",
    "FitsChan",
    "XmlChan",
])

NameMap = {
    "ParallelMap": "CmpMap",
    "SeriesMap": "CmpMap",
}

print()

for name in ExtraIncludes:
    print("#include \"astshim/%s.h\"" % (name,))

for name in name_list:
    if name in SkipNames:
        continue
    print("#include \"astshim/%s.h\"" % (name,))

print()

for name in name_list:
    if name in SkipNames:
        continue
    astName = NameMap.get(name, name)
    print('            {"%s", makeShim<%s, Ast%s>},' % (name, name, astName))

print()

# generate data for CastMap in Object.cc

import glob
import os.path

root_dir = os.path.dirname(os.path.dirname(__file__))
inc_dir = os.path.join(root_dir, "include", "astshim")
path_list = glob.glob(inc_dir + "/*.h")
name_list = [os.path.splitext(os.path.basename(path))[0] for path in path_list]
name_list.sort()

# files to skip (minus trailing ".h")
SkipNames = set([
    "base",
    "makeBadMatrixMap",
    "Stream",
])

# map of astshim name, AST library name without leading "ast"
AstNameMap = {
    "SeriesMap": "CmpMap",
    "ParallelMap": "CmpMap",
}

for name in name_list:
    if name in SkipNames:
        continue
    astName = AstNameMap.get(name, name)
    print '    {"%s", CastAstObject<%s, ast%s>},' % (name, name, astName)

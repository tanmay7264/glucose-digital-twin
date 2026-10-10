"""python run.py                      build twin on 12 synthetic patients, write out/, then serve dashboard
python run.py --serve                serve existing out/ only
python run.py --cgmacros DIR         build on real CGMacros-0XX.csv files in DIR instead (prediabetes+T2D, HbA1c>=5.7)"""
import sys, glob, os
from twin import pipeline

if "--serve" not in sys.argv:
    pats = None
    if "--cgmacros" in sys.argv:
        from twin.data import load_cgmacros_dir
        pats = load_cgmacros_dir(sys.argv[sys.argv.index("--cgmacros") + 1])
    pipeline.main(pats)
from twin.server import serve
serve()

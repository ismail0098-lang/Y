"""The fresh-process PTXAS validator entry point shared with evidence checking.

Keep this module independent of Z3: importing it must not build solver terms.
The exact child program is retained in each command and checked by verify.py.
"""

VALIDATOR_CHILD = """import contextlib, importlib, io, json, sys
sys.path.insert(0, sys.argv[1])
module = importlib.import_module(sys.argv[2])
output = io.StringIO()
with contextlib.redirect_stdout(output):
    try:
        if sys.argv[2] == 'tval':
            verdict, detail, obligations = module.run(
                sys.argv[3], sys.argv[4], NS=6, B1=2, B2=5)
        else:
            verdict, detail, obligations = module.validate(
                sys.argv[3], sys.argv[4], budget=5, mode='wide')
    except Exception as error:
        if 'UNMODELLED' not in str(error) and 'refusing, not guessing' not in str(error):
            raise
        verdict, detail, obligations = 'REFUSED', str(error), 0
print(json.dumps({'format': 'y-ptxas-validation-v1', 'verdict': verdict,
                  'detail': detail, 'obligations': obligations, 'log': output.getvalue()}))
"""

from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
for script in ['generate_data.py','src/part_a.py','src/part_b.py','src/part_c.py','src/part_d.py']:
 print('\n'+'='*20,script,'='*20,flush=True)
 subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)

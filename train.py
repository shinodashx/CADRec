from pathlib import Path
import runpy


if __name__ == '__main__':
    target = Path(__file__).resolve().with_name('train_cadrec.py')
    runpy.run_path(str(target), run_name='__main__')

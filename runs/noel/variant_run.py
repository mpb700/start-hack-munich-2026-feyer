"""Run benchmark.py on a variant fixture made by runs/noel/variant_fixture.py.

    python runs/noel/variant_run.py <repo> <variant folder> [--image IMAGE] [--repeats N] [--keep FOLDER]

benchmark.py reads the two files in `examples/data` and checks their hashes, as a guard against
changed inputs. For a variant this script copies the repo to a temporary folder, puts the
variant files there and enters their hashes into the copy of benchmark.py. Nothing in the repo
changes. With `--image` the benchmark runs in that Docker image with `--cpus 1`, with the variant
files and the adapted benchmark.py mounted over the ones in the image.

Prints the first-run total, the warm median, the sensitivity error against the variant's
reference and the result of the gradient check, and exits with the benchmark's status.
"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

repo, variant = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
option = lambda name, default=None: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
image, repeats, keep = option("--image"), option("--repeats", "3"), option("--keep")
names = ("uifo_800_3000.json", "uifo_800_3000_test_data.json")
hashes = [hashlib.sha256((variant / name).read_bytes()).hexdigest() for name in names]
source = (repo / "benchmark.py").read_text()
for constant, digest in zip(("FIXTURE_SHA256", "REFERENCE_SHA256"), hashes):
    source, count = re.subn(rf'^{constant} = "[0-9a-f]{{64}}"$', f'{constant} = "{digest}"', source, flags=re.M)
    assert count == 1, constant

with tempfile.TemporaryDirectory() as temporary:
    work = Path(temporary)
    if image:
        (work / "benchmark.py").write_text(source)
        (work / "out").mkdir()
        mounts = [f"{work / 'benchmark.py'}:/app/benchmark.py:ro", f"{work / 'out'}:/out"]
        mounts += [f"{variant / name}:/app/examples/data/{name}:ro" for name in names]
        command = ["docker", "run", "--rm", "--cpus", "1"] + [x for mount in mounts for x in ("-v", mount)] + [image, "--repeats", repeats]
        report, cwd = work / "out" / "benchmark-docker.json", None
    else:
        tree = work / "repo"
        shutil.copytree(repo, tree, ignore=shutil.ignore_patterns(".git", ".venv", "runs", "__pycache__", "*.egg-info", "benchmark*.json", "benchmark*.png", "*.npz"))
        (tree / "benchmark.py").write_text(source)
        for name in names:
            shutil.copyfile(variant / name, tree / "examples" / "data" / name)
        report, cwd = work / "report.json", tree
        command = [sys.executable, "benchmark.py", "--repeats", repeats, "--output", str(report)]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if result.returncode != 0 or not report.exists():
        print((result.stdout + result.stderr)[-3000:])
        sys.exit(f"benchmark failed on {variant.name} (exit status {result.returncode})")
    data = json.loads(report.read_text())
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report, Path(keep) / f"{variant.name}{'-docker' if image else ''}.json")

timing, validation = data["timing_seconds"], data["validation"]
check = validation["gradient_check"]
differences = ", ".join(f"{item['central_difference']:.6e}" for item in check["finite_differences"])
print(
    f"{variant.name}{' (Docker, 1 CPU)' if image else ''}: PASS, first run {timing['end_to_end'] * 1e3:.1f} ms, warm {timing['runtime_median'] * 1e3:.1f} ms, "
    f"{data['inputs']['gradient_parameter_count']} design variables, signal {validation['output_shapes']['signal']}"
)
print(
    f"   max |ln s - ln ref| {validation['max_absolute_log_sensitivity_error']:.3e} (limit 1e-5), "
    f"gradient check {check['parameter']}: autodiff {check['autodiff']:.6e}, finite differences {differences}"
)

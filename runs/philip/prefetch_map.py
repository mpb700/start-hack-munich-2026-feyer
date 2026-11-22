"""Record which byte ranges of the installed packages are in the file cache (for differometor/_startup.py, PREFETCH).

    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches          # empty the cache
    python benchmark.py --cpus 1 --repeats 3 ...              # what a run reads is in the cache now
    python runs/philip/prefetch_map.py differometor/_prefetch.bin <site-packages or venv> [...]

Asks the kernel with mincore() which pages of every file below the given directories are in the
cache; mapping a file reads nothing. Only files below a `site-packages` directory are recorded,
by their path relative to it, with their size: on another machine a file is used only if it has
the same size. Linux answers mincore() truthfully only for files the caller owns or may write,
so run this as the owner of the environment. The table is a marshal file:
[(relative path, size, ((offset, length), ...)), ...], largest first.
"""
import ctypes
import marshal
import mmap
import os
import sys

output, roots = sys.argv[1], sys.argv[2:]
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PAGE = mmap.PAGESIZE
FAILED = ctypes.c_void_p(-1).value


def resident_ranges(path, size):
    fd = os.open(path, os.O_RDONLY)
    try:
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if address in (None, FAILED):
            return ()
        pages = (size + PAGE - 1) // PAGE
        vector = (ctypes.c_ubyte * pages)()
        ok = libc.mincore(address, size, vector) == 0
        libc.munmap(address, size)
        if not ok:
            return ()
    finally:
        os.close(fd)
    ranges, start = [], None
    for index in range(pages + 1):
        present = index < pages and vector[index] & 1
        if present and start is None:
            start = index
        elif not present and start is not None:
            ranges.append((start * PAGE, min(index * PAGE, size) - start * PAGE))
            start = None
    return tuple(ranges)


table = []
for root in roots:
    for folder, _, names in os.walk(root):
        if "/site-packages" not in folder + "/":
            continue
        for name in names:
            path = os.path.join(folder, name)
            try:
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                size = os.path.getsize(path)
                ranges = resident_ranges(path, size) if size else ()
            except OSError:
                continue
            if ranges:
                table.append((path.split("/site-packages/", 1)[1], size, ranges))
table.sort(key=lambda entry: -sum(length for _, length in entry[2]))
with open(output, "wb") as file:
    file.write(marshal.dumps(table))
total = sum(length for _, _, ranges in table for _, length in ranges)
print(f"{output}: {len(table)} files, {total / 1e6:.1f} MB in {sum(len(ranges) for _, _, ranges in table)} ranges, {os.path.getsize(output) / 1e3:.0f} kB")
for relative, size, ranges in table[:8]:
    print(f"  {relative}: {sum(length for _, length in ranges) / 1e6:.1f} of {size / 1e6:.1f} MB in {len(ranges)} ranges")

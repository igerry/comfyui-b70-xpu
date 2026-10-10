# -*- coding: utf-8 -*-
"""Patch omni_xpu_kernel/setup.py so the build can use more than one core.

Upstream compiles every core source in ONE icpx invocation (serial inside
clang) and builds the three extensions one after another. This patch:
  1. OMNI_BUILD_JOBS=N  -> setuptools builds the 3 extensions concurrently
  2. core extension     -> per-source `-c` compiles in a thread pool, then a
                           single link step (flags derived from the original
                           cmd list, so nothing else changes)
  3. -fsycl-max-parallel-link-jobs=N on every link/AOT step
  4. OMNI_BUILD_DEVICE_SPLIT=per_source (opt-in) -> lets the AOT step run
                           in parallel by splitting device images per source
"""
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()

# --- 1. parallel extensions -------------------------------------------------
anchor = 'class ICPXBuildExt(build_ext):\n    """Build extension using Intel icpx compiler directly."""\n'
assert anchor in s
s = s.replace(anchor, anchor + '''
    def finalize_options(self):
        super().finalize_options()
        jobs = int(os.environ.get("OMNI_BUILD_JOBS", "1") or "1")
        if jobs > 1:
            # setuptools' build_ext runs build_extension() for each extension
            # in a thread pool when .parallel is set.
            self.parallel = jobs
''')

# --- 2./3./4. per-file parallel compile + parallel link (Linux core only) ---
anchor2 = '        print(f"Compile command: {\' \'.join(cmd)}")\n'
assert anchor2 in s
s = s.replace(anchor2, '''        jobs = int(os.environ.get("OMNI_BUILD_JOBS", "1") or "1")
        split = os.environ.get("OMNI_BUILD_DEVICE_SPLIT", "")
        if not IS_WINDOWS and jobs > 1:
            cmd = _omni_parallel_build(
                cmd, sources, output_path, Path(self.build_temp) / "objs" / ext.name,
                jobs, split, get_compile_env(onednn_include if has_onednn else ""),
                is_single=(is_lgrf or is_cute),
            )
            if cmd is None:
                print(f"Successfully built {output_path} (parallel)")
                return
''' + anchor2)

# helper inserted before the class
helper = '''
def _omni_parallel_build(cmd, sources, output_path, objdir, jobs, split, env, is_single):
    """Split `icpx ... -shared -o out SRCS` into parallel `-c` compiles + one link.

    Returns None when the whole build was done here, or a (possibly modified)
    single-invocation cmd when there is only one source (nothing to split).
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    link_extra = [f"-fsycl-max-parallel-link-jobs={jobs}"]
    if split:
        link_extra.append(f"-fsycl-device-code-split={split}")
    src_set = {str(s) for s in sources}
    if is_single or len(sources) < 2:
        # single-source extensions: only add the parallel link/AOT flags
        return cmd[:1] + link_extra + cmd[1:]

    def is_link_only(tok, prev):
        return (tok in ("-shared",) or tok.startswith("-L") or tok.startswith("-l")
                or tok.startswith("-Wl,") or tok.endswith(".so") or tok.endswith(".a")
                or tok in src_set or tok == str(output_path) or prev == "-o" or tok == "-o")

    compile_base = []
    prev = None
    for tok in cmd:
        if not is_link_only(tok, prev):
            compile_base.append(tok)
        prev = tok
    objdir.mkdir(parents=True, exist_ok=True)

    def compile_one(src):
        obj = objdir / (Path(src).stem + ".o")
        c = compile_base + ["-c", str(src), "-o", str(obj)]
        r = subprocess.run(c, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env)
        return src, obj, r

    print(f"[parallel] compiling {len(sources)} sources with {jobs} jobs")
    objs = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for src, obj, r in ex.map(compile_one, sources):
            if r.returncode != 0:
                print("STDOUT:", r.stdout); print("STDERR:", r.stderr)
                raise RuntimeError(f"Compilation failed for {src}")
            objs.append(str(obj))

    # link: original cmd with sources swapped for objects
    link = [tok for tok in cmd if tok not in src_set]
    link = link[:1] + link_extra + link[1:] + objs
    print(f"[parallel] link: {' '.join(link)}")
    r = subprocess.run(link, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    if r.returncode != 0:
        print("STDOUT:", r.stdout); print("STDERR:", r.stderr)
        raise RuntimeError(f"Link failed with exit code {r.returncode}")
    return None


'''
s = s.replace("class ICPXBuildExt(build_ext):", helper + "class ICPXBuildExt(build_ext):", 1)
open(p, "w", encoding="utf-8").write(s)
print("patched", p)

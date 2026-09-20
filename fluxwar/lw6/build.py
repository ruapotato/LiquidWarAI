"""Build a standalone liblw6ker.so from the Liquid War 6 sources.

Only the kernel is built -- `ker` (the game rules), `map` (the level representation)
and `sys` (portability helpers). `ker` includes nothing outside `map` and `sys`, and
`sys` needs nothing beyond libc, pthread and OpenMP, so the kernel compiles without
any of the game's real dependencies (SDL, guile, curl, ...) and without autotools.

Two files are handled specially:

* `sys-cunit.c` needs CUnit and is only used by LW6's own test binaries; it is skipped.
* `sys-build.c` includes an autotools-generated `sys-build.h` full of build metadata.
  A stand-in is written here; nothing in the kernel reads those values.

`config.h` is generated from `config.h.in` by turning on the standard POSIX headers
and leaving every optional feature off.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

POSIX_HEADERS = """CTYPE_H DIRENT_H ERRNO_H FCNTL_H LANGINFO_H LIMITS_H LOCALE_H MATH_H
PTHREAD_H SIGNAL_H STDARG_H STDIO_H STDLIB_H STRING_H SYSLOG_H SYS_SELECT_H SYS_STAT_H
SYS_SYSINFO_H SYS_TIME_H SYS_TYPES_H SYS_UTSNAME_H TIME_H UNISTD_H EXECINFO_H
INTTYPES_H STDINT_H STRINGS_H MEMORY_H""".split()

VERSION = "0.6.3902"

SYS_BUILD_H = """/* Stand-in for the autotools-generated sys-build.h. Build metadata only. */
#ifndef LIQUIDWAR6SYS_BUILD_H
#define LIQUIDWAR6SYS_BUILD_H
#define LW6_VERSION_BASE "0.6"
#define LW6_VERSION_MAJOR "0"
#define LW6_VERSION_MINOR "6"
#define LW6_CODENAME "standalone-ker"
#define LW6_STAMP "0"
#define LW6_MD5SUM "0"
#define LW6_CONFIGURE_ARGS ""
#define LW6_HOST_CPU "x86_64"
#define LW6_HOST_OS "linux-gnu"
#define LW6_HOSTNAME "standalone"
#define LW6_ABS_SRCDIR ""
#define LW6_TOP_SRCDIR ""
#define LW6_PREFIX "/usr/local"
#define LW6_DATADIR "/usr/local/share"
#define LW6_LIBDIR "/usr/local/lib"
#define LW6_INCLUDEDIR "/usr/local/include"
#define LW6_LOCALEDIR "/usr/local/share/locale"
#define LW6_DOCDIR "/usr/local/share/doc"
#define LW6_CFLAGS ""
#define LW6_LDFLAGS ""
#define LW6_ENABLE_CONSOLE 1
#define LW6_CONSOLE 1
#define LW6_GTK 0
#define LW6_OPENMP 0
#define LW6_ALLINONE 0
#define LW6_FULLSTATIC 0
#define LW6_PARANOID 0
#define LW6_GPROF 0
#define LW6_INSTRUMENT 0
#define LW6_PROFILER 0
#define LW6_GCOV 0
#define LW6_VALGRIND 0
#define LW6_MS_WINDOWS 0
#define LW6_MAC_OS_X 0
#define LW6_X86 0
#define LW6_GP2X 0
/* LW6_UNIX, LW6_GNU, LW6_AMD64 and LW6_OPTIMIZE come from the compiler command line */
#endif
"""


def write_config_h(src_root: Path, out: Path):
    text = (src_root / "config.h.in").read_text()
    lines = []
    for line in text.splitlines():
        m = re.match(r"#undef (\w+)", line)
        if not m:
            lines.append(line)
            continue
        name = m.group(1)
        if name.startswith("HAVE_") and name[5:] in POSIX_HEADERS:
            lines.append(f"#define {name} 1")
        elif name in ("PACKAGE", "PACKAGE_NAME", "PACKAGE_TARNAME"):
            lines.append(f'#define {name} "liquidwar6"')
        elif name in ("VERSION", "PACKAGE_VERSION"):
            lines.append(f'#define {name} "{VERSION}"')
        elif name == "PACKAGE_STRING":
            lines.append(f'#define PACKAGE_STRING "liquidwar6 {VERSION}"')
        elif name == "PACKAGE_BUGREPORT":
            lines.append('#define PACKAGE_BUGREPORT "ufoot@ufoot.org"')
        elif name == "PACKAGE_URL":
            lines.append('#define PACKAGE_URL ""')
        elif name == "STDC_HEADERS":
            lines.append("#define STDC_HEADERS 1")
        elif name == "LW6_OPTIMIZE":
            lines.append("#define LW6_OPTIMIZE 1")
        else:
            lines.append("/* " + line + " */")
    (out / "config.h").write_text("\n".join(lines) + "\n")


def build(src_root: Path, out: Path, jobs: int = 8) -> Path:
    """Compile liblw6ker.so. `src_root` is the liquidwar6/ directory of the sources."""
    src_root = Path(src_root).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (src_root / "src" / "lib" / "ker").is_dir():
        raise FileNotFoundError(f"{src_root} does not look like the liquidwar6 source tree")
    write_config_h(src_root, out)
    (out / "sys-build.h").write_text(SYS_BUILD_H)

    sources = []
    for lib in ("ker", "map", "sys"):
        for c in sorted((src_root / "src" / "lib" / lib).glob("*.c")):
            # Only LW6's own test binaries. Matching "-test" anywhere also drops
            # sys-testandset.c, which is where the lock-free primitive lives, and the
            # library then fails to load with an undefined symbol.
            if c.name.endswith(("-test.c", "-testmain.c")) or c.name == "sys-cunit.c":
                continue
            sources.append(c)
    # The lock-free primitive that -DLW6_AMD64 selects lives in hand-written assembly,
    # not in any .c file.
    sources.append(src_root / "src" / "lib" / "sys" / "sys-testandsetamd64.s")
    # This project's own helpers, compiled in so they can see LW6's struct layouts
    # rather than having Python guess offsets.
    sources.append(Path(__file__).resolve().parent / "csrc" / "fluxwar_helpers.c")
    sources.append(Path(__file__).resolve().parent / "csrc" / "fluxwar_nn.c")
    # mod-nn: the LW6 bot backend that plays with a trained policy. Built here so it
    # can be exercised through LW6's own bot interface without the full game.
    sources.append(Path(__file__).resolve().parent / "csrc" / "mod_nn.c")
    # LW6's own bots, so a trained policy can be benchmarked against the AI the game
    # actually ships with. Only the move and setup units: the backend units exist to
    # register with LW6's dynamic module loader, which is not involved here.
    for mod in ("mod-follow", "mod-brute"):
        for unit in ("move", "setup"):
            sources.append(src_root / "src" / "lib" / "bot" / mod / f"{mod}-{unit}.c")

    # The stub directory supplies ltdl.h and pil.h: LW6's bot headers pull in the
    # dynamic loader and the pilot library, but the bot move code uses neither, so
    # opaque declarations are enough and the real dependencies stay out of the build.
    here = Path(__file__).resolve().parent
    inc = ["-I" + str(out), "-I" + str(here / "stub"), "-I" + str(here / "csrc"),
           "-I" + str(src_root / "src"), "-I" + str(src_root / "src" / "lib")]
    # Deliberately no -fopenmp. The kernel parallelises each team's gradient spread,
    # which is worthless for a single game, and liblw6ker would share libgomp with
    # PyTorch: LW6's spread then runs on torch's OpenMP thread pool, its spinlock
    # debug checks start reporting locks taken from foreign contexts (with garbage
    # function names read from torch's memory), and the process segfaults
    # intermittently. Single-threaded it is both faster here and stable.
    # LW6 passes these on the command line from configure, not through config.h.
    # They are not optional: without -DLW6_AMD64 the spinlocks fall through to a
    # debug mutex path that logs at INFO on every lock, which buried stderr and made
    # a 250-round run take minutes. -DLW6_OPTIMIZE picks the lock-free primitives.
    # LW6_OPTIMIZE compiles out the "bazooka" memory tracker, which otherwise
    # miscounts under this embedding and aborts with "more bytes freed than
    # malloced". Upstream does not actually link with it -- sys-context.c calls
    # _lw6sys_bazooka_context_init unconditionally while sys-bazooka.c only defines
    # it when the flag is off -- so csrc/fluxwar_helpers.c supplies that one stub.
    platform = ["-DLW6_AMD64=1", "-DLW6_UNIX", "-DLW6_GNU", "-DLW6_OPTIMIZE=1"]
    cflags = ["-DHAVE_CONFIG_H", "-O2", "-fPIC", "-w", *platform, *inc]
    subprocess.run(["gcc", "-c", *cflags, *[str(s) for s in sources]],
                   cwd=out, check=True)
    objs = sorted(str(o.name) for o in out.glob("*.o"))
    lib = out / "liblw6ker.so"
    subprocess.run(["gcc", "-shared", "-o", str(lib), *objs, "-lpthread", "-lm"],
                   cwd=out, check=True)
    for o in out.glob("*.o"):
        o.unlink()
    return lib


def clone(dest: Path) -> Path:
    """Shallow-clone the LW6 sources if they are not already there."""
    dest = Path(dest)
    if (dest / "liquidwar6" / "src").is_dir():
        return dest / "liquidwar6"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://git.savannah.gnu.org/git/liquidwar6.git", str(dest)], check=True)
    return dest / "liquidwar6"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="path to the liquidwar6/ source dir")
    ap.add_argument("--out", default="build/lw6")
    a = ap.parse_args()
    src = Path(a.src) if a.src else clone(Path(a.out).parent / "lw6-src")
    print("built:", build(src, Path(a.out)))

#!/usr/bin/env python3
"""
PyTerm (rich) — single-file terminal with colorized output via rich.
Run: python pyterm_rich.py
"""

from __future__ import annotations
import os, sys, shlex, subprocess, shutil, glob, re
from pathlib import Path
from typing import List, Tuple

# Rich imports
from rich import print as rprint
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()

# Optional libs
try:
    import readline
except Exception:
    readline = None

try:
    import psutil
except Exception:
    psutil = None

# State
SHELL_ENV = os.environ.copy()
COMMAND_HISTORY: List[str] = []
HISTORY_FILE = os.path.expanduser("~/.pyterm_history")
MAX_HISTORY = 2000

BUILTIN_NAMES = [
    "cd", "ls", "pwd", "mkdir", "rmdir", "rm", "cat", "touch", "mv", "cp", "glob",
    "export", "history", "help", "exit", "quit", "cpu", "mem", "ps", "top", "run", "nl"
]

# NL patterns
NL_PATTERNS = [
    (re.compile(r"(create|make|new)\s+(a\s+)?(folder|directory)\s+called\s+['\"]?([^'\"]+)['\"]?", re.I),
     lambda m: f"mkdir {m.group(4)}"),
    (re.compile(r"move\s+([^\s]+)\s+(to|into)\s+([^\s]+)", re.I), lambda m: f"mv {m.group(1)} {m.group(3)}"),
    (re.compile(r"(show|display|get)\s+cpu", re.I), lambda m: "cpu"),
    (re.compile(r"(show|display|get)\s+memory", re.I), lambda m: "mem"),
]

def interpret_nl(text: str):
    text = text.strip()
    for patt, fn in NL_PATTERNS:
        m = patt.search(text)
        if m:
            return fn(m)
    return None

# History persistence
def load_history():
    global COMMAND_HISTORY
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            COMMAND_HISTORY = [ln.rstrip("\n") for ln in f.readlines()]
    except Exception:
        COMMAND_HISTORY = []

def save_history():
    try:
        Path(HISTORY_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            for ln in COMMAND_HISTORY[-MAX_HISTORY:]:
                f.write(ln + "\n")
    except Exception:
        pass

# Builtins (same behavior, colorized where helpful)
def builtin_cd(args: List[str]) -> Tuple[int, str]:
    path = args[0] if args else SHELL_ENV.get("HOME", str(Path.home()))
    try:
        os.chdir(os.path.expanduser(path))
        return 0, ""
    except FileNotFoundError:
        return 1, f"[red]cd:[/red] no such file or directory: {path}\n"
    except Exception as e:
        return 1, f"[red]cd error:[/red] {e}\n"

def builtin_ls(args: List[str]) -> Tuple[int, str]:
    path = args[0] if args else "."
    try:
        entries = sorted(os.listdir(path))
        lines = []
        for e in entries:
            p = Path(path) / e
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{e}{suffix}")
        return 0, "\n".join(lines) + ("\n" if lines else "")
    except Exception as e:
        return 1, f"[red]ls error:[/red] {e}\n"

def builtin_pwd(_args: List[str]) -> Tuple[int, str]:
    return 0, f"{os.getcwd()}\n"

def builtin_mkdir(args: List[str]) -> Tuple[int, str]:
    if not args:
        return 1, "[red]mkdir: missing operand[/red]\n"
    out=[]
    for d in args:
        try:
            Path(d).mkdir(parents=True, exist_ok=False)
            out.append(f"[green]created:[/green] {d}")
        except FileExistsError:
            out.append(f"[yellow]mkdir:[/yellow] '{d}' exists")
        except Exception as e:
            out.append(f"[red]mkdir {d}:[/red] {e}")
    return 0, "\n".join(out) + "\n"

def builtin_rm(args: List[str]) -> Tuple[int, str]:
    if not args:
        return 1, "[red]rm: missing operand[/red]\n"
    out=[]
    for p in args:
        try:
            pp = Path(p)
            if pp.is_dir():
                shutil.rmtree(pp)
                out.append(f"[green]removed dir:[/green] {p}")
            else:
                pp.unlink()
                out.append(f"[green]removed file:[/green] {p}")
        except Exception as e:
            out.append(f"[red]rm {p}:[/red] {e}")
    return 0, "\n".join(out) + ("\n" if out else "")

def builtin_cat(args: List[str]) -> Tuple[int, str]:
    if not args:
        return 1, "[red]cat: missing operand[/red]\n"
    out=[]
    for f in args:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                txt = fh.read()
                # use Syntax for highlighting if file is not huge
                out.append(txt)
        except Exception as e:
            out.append(f"[red]cat {f} error:[/red] {e}\n")
    return 0, "".join(out)

def builtin_touch(args: List[str]) -> Tuple[int, str]:
    if not args:
        return 1, "[red]touch: missing operand[/red]\n"
    out=[]
    for f in args:
        try:
            Path(f).touch(exist_ok=True)
            out.append(f"[green]touched:[/green] {f}")
        except Exception as e:
            out.append(f"[red]touch {f}:[/red] {e}")
    return 0, "\n".join(out) + "\n"

def builtin_history(args: List[str]) -> Tuple[int, str]:
    lines = [f"{i:>4}  {c}" for i,c in enumerate(COMMAND_HISTORY, start=1)]
    return 0, "\n".join(lines) + ("\n" if lines else "")

def builtin_export(args: List[str]) -> Tuple[int, str]:
    if not args:
        out="\n".join(f"{k}={v}" for k,v in SHELL_ENV.items())
        return 0, out + ("\n" if out else "")
    for arg in args:
        if "=" in arg:
            k,v = arg.split("=",1)
            SHELL_ENV[k]=v
            os.environ[k]=v
        else:
            return 1, f"[red]export: invalid format:[/red] {arg}\n"
    return 0,""

def builtin_help(_args: List[str]) -> Tuple[int, str]:
    return 0, (
        "Builtins: cd ls pwd mkdir rmdir rm cat touch mv cp glob export history cpu mem ps top run nl help exit\n"
    )

def builtin_cpu(_args: List[str]) -> Tuple[int, str]:
    if psutil is None:
        return 1, "[yellow]cpu: psutil not installed. pip install psutil[/yellow]\n"
    return 0, f"Physical cores: {psutil.cpu_count(logical=False)}\nLogical cores: {psutil.cpu_count(logical=True)}\nTotal CPU%: {psutil.cpu_percent()}%\n"

def builtin_mem(_args: List[str]) -> Tuple[int, str]:
    if psutil is None:
        return 1, "[yellow]mem: psutil not installed. pip install psutil[/yellow]\n"
    vm = psutil.virtual_memory()
    return 0, f"Total: {vm.total/1024**3:.2f} GB\nUsed: {vm.used/1024**3:.2f} GB ({vm.percent}%)\n"

def builtin_ps(_args: List[str]) -> Tuple[int, str]:
    if psutil is None:
        return 1, "[yellow]ps: psutil not installed. pip install psutil[/yellow]\n"
    rows=[]
    for p in psutil.process_iter(['pid','name','cpu_percent','memory_percent']):
        try:
            info=p.info
            rows.append(f"{info.get('pid')}\t{info.get('name','')}\tCPU%:{info.get('cpu_percent')}\tMem%:{info.get('memory_percent')}")
        except Exception:
            continue
    return 0, "\n".join(rows) + ("\n" if rows else "")

def builtin_top(args: List[str]) -> Tuple[int, str]:
    if psutil is None:
        return 1, "[yellow]top: psutil not installed. pip install psutil[/yellow]\n"
    n = int(args[0]) if args and args[0].isdigit() else 5
    procs=[]
    for p in psutil.process_iter(['pid','name','cpu_percent']):
        try:
            procs.append((p.pid, p.info.get('name',''), p.info.get('cpu_percent',0)))
        except Exception:
            continue
    procs=sorted(procs, key=lambda x: x[2] or 0, reverse=True)[:n]
    out=[f"{pid}\t{name}\tCPU%:{cpu}" for pid,name,cpu in procs]
    return 0, "\n".join(out) + ("\n" if out else "")

def builtin_mv(args: List[str]) -> Tuple[int, str]:
    if len(args) < 2:
        return 1, "[red]mv: missing operand[/red]\n"
    out=[]
    dst=args[-1]
    for s in args[:-1]:
        try:
            shutil.move(s, dst)
            out.append(f"[green]moved:[/green] {s} -> {dst}")
        except Exception as e:
            out.append(f"[red]mv {s} {dst}:[/red] {e}")
    return 0, "\n".join(out) + "\n"

def builtin_cp(args: List[str]) -> Tuple[int, str]:
    if len(args) < 2:
        return 1, "[red]cp: missing operand[/red]\n"
    out=[]
    dst=args[-1]
    for s in args[:-1]:
        try:
            p=Path(s)
            if p.is_dir():
                dest=Path(dst)/p.name if Path(dst).is_dir() else Path(dst)
                shutil.copytree(s,dest)
            else:
                if Path(dst).is_dir():
                    shutil.copy2(s, Path(dst)/p.name)
                else:
                    shutil.copy2(s, dst)
            out.append(f"[green]copied:[/green] {s} -> {dst}")
        except Exception as e:
            out.append(f"[red]cp {s} {dst}:[/red] {e}")
    return 0, "\n".join(out) + "\n"

def builtin_glob(args: List[str]) -> Tuple[int, str]:
    if not args:
        return 1, "[red]glob: missing operand[/red]\n"
    out=[]
    for pat in args:
        out.extend(glob.glob(pat))
    return 0, "\n".join(out) + ("\n" if out else "")

BUILTINS = {
    "cd": builtin_cd, "ls": builtin_ls, "pwd": builtin_pwd, "mkdir": builtin_mkdir,
    "rmdir": builtin_rmdir, "rm": builtin_rm, "cat": builtin_cat, "touch": builtin_touch,
    "mv": builtin_mv, "cp": builtin_cp, "glob": builtin_glob, "history": builtin_history,
    "export": builtin_export, "help": builtin_help, "cpu": builtin_cpu, "mem": builtin_mem,
    "ps": builtin_ps, "top": builtin_top
}

# Execution helpers (same as before)
def run_external(parts: List[str], input_data: str = None, use_shell: bool = False) -> Tuple[int, str]:
    try:
        if use_shell:
            cmd_str = " ".join(parts)
            res = subprocess.run(cmd_str, capture_output=True, text=True, shell=True, env=SHELL_ENV, input=input_data)
        else:
            res = subprocess.run(parts, capture_output=True, text=True, shell=False, env=SHELL_ENV, input=input_data)
        out = (res.stdout or "") + (res.stderr or "")
        return res.returncode, out
    except FileNotFoundError:
        return 127, f"{parts[0]}: command not found\n"
    except Exception as e:
        return 1, f"error running {parts[0]}: {e}\n"

def execute_builtin_or_external(parts: List[str], input_data: str = None) -> Tuple[int, str]:
    if not parts:
        return 0,""
    cmd=parts[0]
    args=[os.path.expandvars(a) for a in parts[1:]]
    if cmd == "run":
        if not args:
            return 1, "[red]run: missing command[/red]\n"
        return run_external(args, input_data=input_data, use_shell=True)
    if cmd in BUILTINS:
        try:
            return BUILTINS[cmd](args)
        except Exception as e:
            return 1, f"[red]builtin {cmd} error:[/red] {e}\n"
    return run_external([cmd]+args, input_data=input_data, use_shell=False)

def handle_pipeline_and_redirection(user_input: str) -> Tuple[int, str]:
    s=user_input.strip()
    segments=[seg.strip() for seg in s.split("|")]
    prev_output=None
    prev_code=0
    out_redirect=None
    last_seg=segments[-1]
    if ">>" in last_seg:
        parts=last_seg.rsplit(">>",1)
        segments[-1]=parts[0].strip()
        out_redirect=(parts[1].strip(),"a")
    elif ">" in last_seg:
        parts=last_seg.rsplit(">",1)
        segments[-1]=parts[0].strip()
        out_redirect=(parts[1].strip(),"w")
    for seg in segments:
        if not seg:
            return 1,"[red]bad null command in pipeline[/red]\n"
        try:
            parts=shlex.split(seg)
        except Exception as e:
            return 1, f"[red]parse error:[/red] {e}\n"
        if parts[0] in BUILTINS:
            prev_code, prev_output = execute_builtin_or_external(parts)
        else:
            prev_code, prev_output = execute_builtin_or_external(parts, input_data=prev_output)
    if prev_output is None:
        return prev_code,""
    if out_redirect:
        fn,mode=out_redirect
        try:
            with open(fn, mode, encoding="utf-8") as fh:
                fh.write(prev_output)
            return 0,""
        except Exception as e:
            return 1, f"[red]redirection error:[/red] {e}\n"
    return prev_code, prev_output

# Tab completion
def completer(text, state):
    suggestions=[c for c in BUILTIN_NAMES if c.startswith(text)]
    try:
        file_matches=glob.glob(text+"*")
    except Exception:
        file_matches=[]
    candidates=suggestions+file_matches
    try:
        return candidates[state]
    except IndexError:
        return None

def init_readline():
    if not readline:
        return
    try:
        readline.read_history_file(HISTORY_FILE)
    except Exception:
        pass
    readline.set_history_length(1000)
    try:
        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

# REPL
def repl():
    load_history()
    init_readline()
    rprint("[bold blue]PyTerm[/bold blue] — type 'help'. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            cwd=os.getcwd()
            prompt=f"[green]py-term:[/green]{cwd} $ "
            if readline:
                user_input = input(prompt)
            else:
                user_input = input(prompt)
        except KeyboardInterrupt:
            rprint("")  # newline
            continue
        except EOFError:
            rprint("[bold]exit[/bold]")
            save_history()
            break
        if not user_input.strip():
            continue
        COMMAND_HISTORY.append(user_input)
        if len(COMMAND_HISTORY)>MAX_HISTORY:
            COMMAND_HISTORY[:] = COMMAND_HISTORY[-MAX_HISTORY:]
        save_history()
        line=user_input.strip()
        mapped=None
        if line.lower().startswith("nl "):
            nltext=line[3:].strip().strip('"').strip("'")
            mapped=interpret_nl(nltext)
            if mapped:
                rprint(f"[cyan]interpreted:[/cyan] {mapped}")
                line=mapped
        elif (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            nltext=line.strip().strip('"').strip("'")
            mapped=interpret_nl(nltext)
            if mapped:
                rprint(f"[cyan]interpreted:[/cyan] {mapped}")
                line=mapped
        if line in ("exit","quit"):
            rprint("[bold]Goodbye![/bold]")
            save_history()
            break
        if "|" in line or ">" in line or ">>" in line:
            code,out = handle_pipeline_and_redirection(line)
            if out:
                console.print(out, end="")
            continue
        try:
            parts=shlex.split(line)
        except Exception as e:
            rprint(f"[red]parse error:[/red] {e}")
            continue
        if not parts:
            continue
        code,out = execute_builtin_or_external(parts)
        if out:
            console.print(out, end="")

if __name__ == "__main__":
    repl()

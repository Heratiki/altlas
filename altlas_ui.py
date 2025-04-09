from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
import subprocess
import sys

console = Console()


def list_benchmarks():
    bench_dir = Path('task/benchmarks')
    if not bench_dir.exists():
        console.print('[red]Benchmark directory not found.[/red]')
        return []
    return sorted(bench_dir.glob('benchmark_*.json'))


def view_latest_report():
    report_path = Path('memory/reports/latest_report.md')
    if report_path.exists():
        console.rule("Latest Training Report")
        content = report_path.read_text(encoding='utf-8')
        console.print(content)
    else:
        console.print('[yellow]No latest report found.[/yellow]')


def main():
    console.rule("[bold green]AltLAS Terminal UI[/bold green]")

    benchmarks = list_benchmarks()
    if not benchmarks:
        console.print("[red]No benchmarks found. Exiting.[/red]")
        sys.exit(1)

    # Display all options in one menu
    console.print("[bold cyan]Available Tasks:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#")
    table.add_column("Benchmark File")
    for idx, bm in enumerate(benchmarks, 1):
        table.add_row(str(idx), bm.name)
    console.print(table)

    console.print("\n[bold cyan]Options:[/bold cyan]")
    console.print("[yellow]V[/yellow]: View latest training report")
    console.print("[yellow]R[/yellow]: Toggle reset flag")
    console.print("[yellow]D[/yellow]: Toggle debug mode")
    console.print("[yellow]Q[/yellow]: Quit")

    reset_flag = False
    debug_flag = False
    selected_task = None

    while True:
        choice = Prompt.ask("Enter task number or option (V/R/D/Q)").strip().lower()
        if choice == 'q':
            console.print("Exiting.")
            sys.exit(0)
        elif choice == 'v':
            view_latest_report()
        elif choice == 'r':
            reset_flag = not reset_flag
            console.print(f"Reset flag is now [{'ON' if reset_flag else 'OFF'}]")
        elif choice == 'd':
            debug_flag = not debug_flag
            console.print(f"Debug mode is now [{'ON' if debug_flag else 'OFF'}]")
        elif choice.isdigit() and 1 <= int(choice) <= len(benchmarks):
            selected_task = benchmarks[int(choice)-1]
            break
        else:
            console.print("[red]Invalid input. Please try again.[/red]")

    # Prepare task argument
    task_arg = selected_task.name
    if task_arg.endswith('.json'):
        task_arg = task_arg[:-5]

    options = []
    if reset_flag:
        options.append('--reset')
    if debug_flag:
        options.append('--debug')

    cmd = [sys.executable, 'runner.py', '--task', task_arg] + options

    console.print(f"\n[bold]Launching:[/bold] {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Runner exited with error code {e.returncode}[/red]")


if __name__ == "__main__":
    main()

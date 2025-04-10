#!/usr/bin/env python3
"""
AltLAS UIDisplay - Module for handling the Rich UI components and display.
"""

import time
import logging
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

log = logging.getLogger(__name__)

class UIDisplay:
    """
    Handles the Rich UI components and display for AltLAS.
    """
    def __init__(self):
        """Initialize the UI display components."""
        # Create separate console objects for logging and for the Live display
        self.log_console = Console(stderr=True)  # Use stderr for logging to avoid conflicts
        self.ui_console = Console()  # Use stdout for the Live UI display
        self.ui_console.quiet = False  # Ensure signals are not swallowed by the Live display console
        
        # Create the Rich layout
        self.layout = Layout()
        self.layout.split(
            Layout(name="header", size=10), # Increased size for progress and timing info
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="status", ratio=2)
        )
        
        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            TaskProgressColumn(),
            TextColumn("[bold]{task.fields[percentage]:.2f}%")
        )
        
        # Store UI state
        self.status_messages = []
        self.task_id = None
        self.live = None
        self.start_time = None
        self.attempt_times = []  # For rate calculation
        self.RATE_WINDOW = 100  # Number of attempts to average for rate calculation
    
    def initialize(self, max_attempts, current_task_name):
        """
        Initialize the UI display with task information.
        
        Args:
            max_attempts (int): The maximum number of attempts
            current_task_name (str): The name of the current task
        """
        self.start_time = time.time()
        self.task_id = self.progress.add_task(
            f"[blue]Running task: {current_task_name}", 
            total=max_attempts, 
            percentage=0
        )
        
        # Initialize UI
        self._update_header(0, max_attempts)
        self._update_stats_panel({
            "Total Attempts": 0,
            "Success Attempts": 0,
            "Error Attempts": 0,
            "Duplicate Attempts": 0,
            "Unsafe Attempts": 0,
            "Hints Requested": 0,
            "Hints Provided": 0
        })
        self._update_status_panel()
        self._update_footer_panel(None)
        
        log.info(f"UI initialized for task: {current_task_name}, max attempts: {max_attempts}")
        
        return self.task_id
    
    def start(self):
        """
        Start the Live display.
        
        Returns:
            rich.live.Live: The Live object for context management
        """
        self.live = Live(
            self.layout, 
            refresh_per_second=2,  # Reduced refresh rate from 4 to 2
            console=self.ui_console
        )
        return self.live
    
    def add_status_message(self, message):
        """
        Add a status message to the display.
        
        Args:
            message (str): The message to add
        """
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        self.status_messages.insert(0, formatted_message)
        if len(self.status_messages) > 100:  # Limit the number of status messages
            self.status_messages = self.status_messages[:100]
        
        # Update the status panel if live display is active
        if self.live and self.live.is_started:
            self._update_status_panel()
    
    def update_attempt(self, attempt_count, max_attempts, stats=None, hint=None):
        """
        Update the UI with the latest attempt information.
        
        Args:
            attempt_count (int): The current attempt count
            max_attempts (int): The maximum number of attempts
            stats (dict, optional): Dictionary of statistics to display
            hint (str, optional): The current hint, if any
        """
        # Record attempt time for rate calculation
        current_time = time.time()
        self.attempt_times.append(current_time)
        if len(self.attempt_times) > self.RATE_WINDOW:
            self.attempt_times.pop(0)
        
        # Update header with progress and timing
        self._update_header(attempt_count, max_attempts)
        
        # Update stats panel if stats provided
        if stats:
            self._update_stats_panel(stats)
        
        # Update footer with hint if provided
        self._update_footer_panel(hint)
    
    def _update_header(self, attempt_count, max_attempts):
        """Update the header panel with progress and timing information."""
        # Update progress bar
        self.progress.update(
            self.task_id, 
            completed=attempt_count, 
            percentage=100*attempt_count/max_attempts
        )
        
        # Calculate timing metrics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate attempt rate
        if len(self.attempt_times) >= 2:
            current_rate = len(self.attempt_times) / (self.attempt_times[-1] - self.attempt_times[0])
        else:
            current_rate = 0
        
        # Calculate estimated time remaining
        remaining_attempts = max_attempts - attempt_count
        if current_rate > 0:
            estimated_time_remaining = remaining_attempts / current_rate
        else:
            estimated_time_remaining = float('inf')
        # Calculate estimated total time to 100%
        progress_percent = 100 * attempt_count / max_attempts if max_attempts > 0 else 0
        if progress_percent > 0:
            total_estimated_time = elapsed_time / (progress_percent / 100)
        else:
            total_estimated_time = float('inf')
        
        # Create timing table
        timing_table = Table(show_header=False, box=None)
        timing_table.add_row("Est. Total Time", self._format_time(total_estimated_time))
        timing_table.add_row("Elapsed Time", self._format_time(elapsed_time))
        timing_table.add_row("Est. Remaining", self._format_time(estimated_time_remaining))
        timing_table.add_row("Current Rate", f"{current_rate:.1f} attempts/sec")
        timing_table.add_row("Avg Time/Attempt", 
                           f"{(elapsed_time/max(1, attempt_count)):.2f}s")
        
        # Create a new layout for the header content
        header_content = Layout()
        header_content.split(
            Layout(self.progress, size=1), # Progress bar typically needs only 1 line
            Layout(timing_table)
        )
        
        # Update header panel
        header_panel = Panel(
            header_content,
            title="AltLAS Progress",
            border_style="blue"
        )
        self.layout["header"].update(header_panel)
    
    def _update_stats_panel(self, stats):
        """Update the stats panel with the provided statistics."""
        stats_table = Table(show_header=False, box=None)
        
        # Add basic stats
        for key, value in stats.items():
            if key == "Best Code":  # Special handling for code display
                continue  # Skip in this loop to add at the end with special formatting
            elif key in ["Highest Score", "Current Entropy Coef"]:
                # Format floating point values
                if isinstance(value, float):
                    stats_table.add_row(key, f"{value:.4f}")
                else:
                    stats_table.add_row(key, str(value))
            else:
                stats_table.add_row(key, str(value))
        
        # Add best code if present
        if "Best Code" in stats and stats["Best Code"]:
            # Truncate long code for display
            code = stats["Best Code"]
            if len(code) > 100:
                code = code[:97] + "..."
            stats_table.add_row("Best Code", code)
        
        self.layout["stats"].update(Panel(
            stats_table, 
            title="Statistics", 
            border_style="green"
        ))
    
    def _update_status_panel(self):
        """Update the status panel with current messages."""
        status_panel = Text("\n".join(self.status_messages[:15]))
        self.layout["status"].update(Panel(
            status_panel, 
            title="Status Messages", 
            border_style="yellow"
        ))
    
    def _update_footer_panel(self, hint):
        """Update the footer panel with the current hint."""
        if hint:
            hint_text = Text(f"🤔 Current Hint: {hint}", style="italic yellow")
            self.layout["footer"].update(Panel(
                hint_text, 
                title="Advisor Hint", 
                border_style="yellow"
            ))
        else:
            self.layout["footer"].update(Panel(
                "No active hints", 
                title="Advisor Hint", 
                border_style="dim"
            ))
    
    def _format_time(self, seconds):
        """Format seconds into a human-readable time string."""
        if seconds == float('inf'):
            return "∞"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def print_summary(self, success, user_interrupted, attempt_count, max_attempts, 
                     best_score=None, best_attempt=None, run_stats=None):
        """
        Print a summary of the run at the end.
        
        Args:
            success (bool): Whether the task was completed successfully
            user_interrupted (bool): Whether the run was interrupted by the user
            attempt_count (int): The final attempt count
            max_attempts (int): The maximum number of attempts
            best_score (float, optional): The best score achieved
            best_attempt (int, optional): The attempt number of the best score
            run_stats (dict, optional): Additional run statistics to display
        """
        summary = []
        
        # Add status message
        if success:
            summary.append("[bold green]✅ Task completed successfully![/bold green]")
        elif user_interrupted:
            summary.append("[bold yellow]🏃 Run interrupted by user (Ctrl+C).[/bold yellow]")
        elif attempt_count >= max_attempts:
            summary.append("[bold yellow]⏱️ Maximum attempts reached.[/bold yellow]")
        else:
            summary.append("[bold red]⏹️ Run ended unexpectedly.[/bold red]")
        
        # Add best score information
        if best_score is not None and best_attempt is not None:
            summary.append(f"[bold blue]📈 Best score achieved: {best_score:.2f}[/bold blue] (Attempt {best_attempt})")
        
        # Add run statistics
        if run_stats:
            summary.append("\n[bold]Run Statistics:[/bold]")
            for key, value in run_stats.items():
                summary.append(f"  - {key}: {value}")
        
        # Print summary
        self.ui_console.print(Panel("\n".join(summary), title="Run Summary", border_style="bold"))
    
    def print_token_frequencies(self, token_frequency):
        """
        Print a table of token frequencies.
        
        Args:
            token_frequency (dict): Dictionary mapping tokens to their frequencies
        """
        if token_frequency:
            self.ui_console.print("[bold blue]Final Generated Token Frequencies:[/bold blue]")
            
            # Sort for consistent output
            sorted_final_freq = sorted(token_frequency.items(), key=lambda item: item[1], reverse=True)
            
            # Use Rich Table for better formatting
            freq_table = Table(title="Token Frequencies", show_header=True, header_style="bold magenta")
            freq_table.add_column("Token", style="dim")
            freq_table.add_column("Count", justify="right")
            
            for token, count in sorted_final_freq:
                # Escape special characters like newlines for display
                display_token = repr(token).strip("'")
                freq_table.add_row(display_token, str(count))
                
            self.ui_console.print(freq_table)
        else:
            self.ui_console.print("[yellow]No token frequency data collected.[/yellow]")
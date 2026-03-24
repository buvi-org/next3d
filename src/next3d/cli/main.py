"""CLI entry point for next3d.

Commands:
    next3d inspect <file.step>     — Print semantic summary
    next3d graph <file.step>       — Export full semantic graph as JSON
    next3d query <file.step> "..." — Run DSL query, print results
    next3d features <file.step>    — List recognized features
    next3d validate <file.step>    — Check file integrity
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from next3d.graph.semantic import build_semantic_graph
from next3d.graph.query import execute_query
from next3d.ai.interface import to_json, to_summary

console = Console()


def _load_graph(step_file: str):
    """Load and build semantic graph, handling errors."""
    path = Path(step_file)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        sys.exit(1)
    try:
        return build_semantic_graph(path)
    except Exception as e:
        console.print(f"[red]Error processing STEP file:[/red] {e}")
        sys.exit(1)


@click.group()
@click.version_option(package_name="next3d")
def cli():
    """Next3D — Semantic 3D Geometry Cognition System."""


@cli.command()
@click.argument("step_file")
def inspect(step_file: str):
    """Print a semantic summary of a STEP file."""
    graph = _load_graph(step_file)
    summary = to_summary(graph)

    console.print(f"\n[bold]Next3D Inspection: {step_file}[/bold]\n")

    # Statistics table
    table = Table(title="Statistics")
    table.add_column("Entity", style="cyan")
    table.add_column("Count", style="green", justify="right")
    for k, v in summary["statistics"].items():
        table.add_row(k.capitalize(), str(v))
    console.print(table)

    # Face distribution
    if summary["face_type_distribution"]:
        dt = Table(title="Face Types")
        dt.add_column("Surface Type", style="cyan")
        dt.add_column("Count", style="green", justify="right")
        for k, v in summary["face_type_distribution"].items():
            dt.add_row(k, str(v))
        console.print(dt)

    # Features
    if summary["features"]:
        ft = Table(title="Recognized Features")
        ft.add_column("ID", style="dim")
        ft.add_column("Type", style="yellow")
        ft.add_column("Description", style="white")
        ft.add_column("Parameters", style="cyan")
        for f in summary["features"]:
            ft.add_row(
                f["id"][:16] + "...",
                f["type"],
                f["description"],
                json.dumps(f["parameters"]),
            )
        console.print(ft)
    else:
        console.print("[dim]No features recognized.[/dim]")


@cli.command()
@click.argument("step_file")
@click.option("--mode", type=click.Choice(["summary", "detail"]), default="detail")
@click.option("--output", "-o", type=click.Path(), help="Write to file instead of stdout")
def graph(step_file: str, mode: str, output: str | None):
    """Export semantic graph as JSON."""
    g = _load_graph(step_file)
    result = to_json(g, mode=mode)

    if output:
        Path(output).write_text(result)
        console.print(f"[green]Written to {output}[/green]")
    else:
        click.echo(result)


@cli.command()
@click.argument("step_file")
@click.argument("query_str")
def query(step_file: str, query_str: str):
    """Run a DSL query against a STEP file."""
    g = _load_graph(step_file)
    try:
        result = execute_query(g, query_str)
    except ValueError as e:
        console.print(f"[red]Query error:[/red] {e}")
        sys.exit(1)

    console.print(f"\n[bold]Query:[/bold] {query_str}")
    console.print(f"[bold]Results:[/bold] {len(result)} entities\n")

    for entity in result:
        console.print_json(entity.model_dump_json())


@cli.command()
@click.argument("step_file")
def features(step_file: str):
    """List recognized features with parameters."""
    g = _load_graph(step_file)

    if not g.features:
        console.print("[dim]No features recognized.[/dim]")
        return

    table = Table(title=f"Features in {step_file}")
    table.add_column("ID", style="dim")
    table.add_column("Type", style="yellow")
    table.add_column("Faces", style="cyan", justify="right")
    table.add_column("Parameters", style="white")
    table.add_column("Description")

    for f in g.features:
        table.add_row(
            f.persistent_id[:16] + "...",
            f.feature_type.value,
            str(len(f.face_ids)),
            json.dumps(f.parameters),
            f.description,
        )
    console.print(table)


@cli.command()
@click.argument("step_file")
def validate(step_file: str):
    """Check STEP file integrity and report issues."""
    path = Path(step_file)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        sys.exit(1)

    try:
        g = build_semantic_graph(path)
    except Exception as e:
        console.print(f"[red]INVALID:[/red] {e}")
        sys.exit(1)

    issues = []
    if not g.solids:
        issues.append("No solids found")
    if not g.faces:
        issues.append("No faces found")

    # Check for isolated faces (no adjacency)
    faces_with_adj = set()
    for adj in g.adjacency:
        faces_with_adj.add(adj.source_id)
        faces_with_adj.add(adj.target_id)
    isolated = [f for f in g.faces if f.persistent_id not in faces_with_adj]
    if isolated:
        issues.append(f"{len(isolated)} isolated face(s) with no adjacency")

    if issues:
        console.print(f"[yellow]WARNINGS ({len(issues)}):[/yellow]")
        for issue in issues:
            console.print(f"  - {issue}")
    else:
        console.print("[green]VALID[/green] — No issues found")

    console.print(f"\nSolids: {len(g.solids)}, Faces: {len(g.faces)}, "
                  f"Edges: {len(g.edges)}, Features: {len(g.features)}")

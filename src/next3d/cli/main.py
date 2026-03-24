"""CLI entry point for next3d.

Commands:
    next3d inspect <file.step>     — Print semantic summary
    next3d graph <file.step>       — Export full semantic graph as JSON
    next3d query <file.step> "..." — Run DSL query, print results
    next3d features <file.step>    — List recognized features
    next3d validate <file.step>    — Check file integrity
    next3d properties <file.step>  — Physical properties (mass, CoG, inertia)
    next3d manufacturing <file.step> — Manufacturing analysis
    next3d embeddings <file.step>  — Graph embedding export for ML
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


def _format_output(data: dict | list, fmt: str) -> str:
    """Format output data as json, table, or yaml."""
    if fmt == "json":
        return json.dumps(data, indent=2)
    elif fmt == "yaml":
        # Simple YAML-like output without requiring pyyaml
        return _dict_to_yaml(data)
    else:
        return json.dumps(data, indent=2)


def _dict_to_yaml(obj, indent: int = 0) -> str:
    """Minimal YAML serializer (no external dependency)."""
    lines: list[str] = []
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{prefix}{k}:")
                lines.append(_dict_to_yaml(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {_yaml_value(v)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_dict_to_yaml(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_yaml_value(item)}")
    else:
        lines.append(f"{prefix}{_yaml_value(obj)}")
    return "\n".join(lines)


def _yaml_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"' if any(c in v for c in ":#{}[]&*!|>'\"%@`") else v
    return str(v)


@click.group()
@click.version_option(package_name="next3d")
def cli():
    """Next3D — Semantic 3D Geometry Cognition System."""


@cli.command()
@click.argument("step_file")
@click.option("--format", "fmt", type=click.Choice(["json", "table", "yaml"]), default="table")
def inspect(step_file: str, fmt: str):
    """Print a semantic summary of a STEP file."""
    graph = _load_graph(step_file)
    summary = to_summary(graph)

    if fmt in ("json", "yaml"):
        click.echo(_format_output(summary, fmt))
        return

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

    # Relationships
    if summary.get("relationships"):
        rt = Table(title="Relationships")
        rt.add_column("Type", style="yellow")
        rt.add_column("Count", style="green", justify="right")
        rel_counts: dict[str, int] = {}
        for r in graph.relationships:
            rel_counts[r.relationship_type.value] = rel_counts.get(r.relationship_type.value, 0) + 1
        for rtype, count in sorted(rel_counts.items()):
            rt.add_row(rtype, str(count))
        console.print(rt)


@cli.command()
@click.argument("step_file")
@click.option("--mode", type=click.Choice(["summary", "detail", "stream"]), default="detail")
@click.option("--output", "-o", type=click.Path(), help="Write to file instead of stdout")
@click.option("--format", "fmt", type=click.Choice(["json", "yaml"]), default="json")
def graph(step_file: str, mode: str, output: str | None, fmt: str):
    """Export semantic graph as JSON.

    Modes: summary (compact), detail (full), stream (NDJSON line-by-line).
    """
    g = _load_graph(step_file)

    if mode == "stream":
        from next3d.ai.streaming import stream_semantic_graph
        if output:
            with open(output, "w") as f:
                for line in stream_semantic_graph(g):
                    f.write(line + "\n")
            console.print(f"[green]Streamed to {output}[/green]")
        else:
            for line in stream_semantic_graph(g):
                click.echo(line)
        return

    if fmt == "json":
        result = to_json(g, mode=mode)
    else:
        from next3d.ai.interface import to_summary as _sum, to_detail as _det
        data = _sum(g) if mode == "summary" else _det(g)
        result = _format_output(data, fmt)

    if output:
        Path(output).write_text(result)
        console.print(f"[green]Written to {output}[/green]")
    else:
        click.echo(result)


@cli.command()
@click.argument("step_file")
@click.argument("query_str")
@click.option("--format", "fmt", type=click.Choice(["json", "table", "yaml"]), default="table")
def query(step_file: str, query_str: str, fmt: str):
    """Run a DSL query against a STEP file."""
    g = _load_graph(step_file)
    try:
        result = execute_query(g, query_str)
    except ValueError as e:
        console.print(f"[red]Query error:[/red] {e}")
        sys.exit(1)

    if fmt in ("json", "yaml"):
        entities = [json.loads(e.model_dump_json()) for e in result]
        click.echo(_format_output(entities, fmt))
        return

    console.print(f"\n[bold]Query:[/bold] {query_str}")
    console.print(f"[bold]Results:[/bold] {len(result)} entities\n")

    for entity in result:
        console.print_json(entity.model_dump_json())


@cli.command()
@click.argument("step_file")
@click.option("--format", "fmt", type=click.Choice(["json", "table", "yaml"]), default="table")
def features(step_file: str, fmt: str):
    """List recognized features with parameters."""
    g = _load_graph(step_file)

    if fmt in ("json", "yaml"):
        data = [json.loads(f.model_dump_json()) for f in g.features]
        click.echo(_format_output(data, fmt))
        return

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


@cli.command()
@click.argument("step_file")
@click.option("--material", default="steel", help="Material name or density in g/mm³")
@click.option("--format", "fmt", type=click.Choice(["json", "table", "yaml"]), default="table")
def properties(step_file: str, material: str, fmt: str):
    """Compute physical properties (mass, CoG, moments of inertia)."""
    from next3d.core.brep import load_step
    from next3d.core.properties import MATERIALS, compute_physical_properties

    path = Path(step_file)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        sys.exit(1)

    model = load_step(path)

    # Resolve density
    try:
        density = float(material)
    except ValueError:
        density = MATERIALS.get(material.lower())
        if density is None:
            console.print(f"[red]Unknown material:[/red] {material}. "
                          f"Available: {', '.join(MATERIALS.keys())}")
            sys.exit(1)

    props = compute_physical_properties(model.shape, density=density)

    if fmt in ("json", "yaml"):
        click.echo(_format_output(props.to_dict(), fmt))
        return

    console.print(f"\n[bold]Physical Properties: {step_file}[/bold]")
    console.print(f"Material density: {density} g/mm³\n")

    table = Table(title="Properties")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Unit", style="dim")

    table.add_row("Volume", f"{props.volume:.2f}", "mm³")
    table.add_row("Surface Area", f"{props.surface_area:.2f}", "mm²")
    table.add_row("Mass", f"{props.mass:.4f}", "g")
    cog = props.center_of_gravity
    table.add_row("Center of Gravity", f"({cog.x:.3f}, {cog.y:.3f}, {cog.z:.3f})", "mm")
    table.add_row("Ixx", f"{props.ixx:.2f}", "g·mm²")
    table.add_row("Iyy", f"{props.iyy:.2f}", "g·mm²")
    table.add_row("Izz", f"{props.izz:.2f}", "g·mm²")
    console.print(table)


@cli.command()
@click.argument("step_file")
@click.option("--format", "fmt", type=click.Choice(["json", "table", "yaml"]), default="table")
def manufacturing(step_file: str, fmt: str):
    """Analyze manufacturing requirements."""
    from next3d.core.manufacturing import analyze_manufacturing

    g = _load_graph(step_file)
    analysis = analyze_manufacturing(g)

    if fmt in ("json", "yaml"):
        click.echo(_format_output(analysis.to_dict(), fmt))
        return

    console.print(f"\n[bold]Manufacturing Analysis: {step_file}[/bold]\n")

    # Summary
    console.print(f"Minimum axes required: [bold]{analysis.min_axes}[/bold]-axis")
    console.print(f"Complexity score: [bold]{analysis.complexity_score:.0f}[/bold] / 100")
    console.print(f"Suggested processes: {', '.join(analysis.suggested_processes)}\n")

    # Machining axes
    if analysis.machining_axes:
        at = Table(title="Machining Axes")
        at.add_column("Direction", style="cyan")
        at.add_column("Features", style="green", justify="right")
        at.add_column("Description")
        for ax in analysis.machining_axes:
            d = ax.direction
            at.add_row(f"({d.x:.2f}, {d.y:.2f}, {d.z:.2f})", str(len(ax.features)), ax.description)
        console.print(at)

    # Feature assessments
    if analysis.feature_assessments:
        ft = Table(title="Feature Manufacturability")
        ft.add_column("Type", style="yellow")
        ft.add_column("Process", style="cyan")
        ft.add_column("Difficulty", style="green")
        ft.add_column("Tool")
        for a in analysis.feature_assessments:
            ft.add_row(
                a.get("feature_type", ""),
                a.get("process", ""),
                a.get("difficulty", ""),
                a.get("tool", ""),
            )
        console.print(ft)

    # Warnings
    if analysis.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in analysis.warnings:
            console.print(f"  - {w}")


@cli.command()
@click.argument("step_file")
@click.option("--output", "-o", type=click.Path(), help="Write to file")
def embeddings(step_file: str, output: str | None):
    """Export graph embeddings for GNN/ML models."""
    from next3d.ai.embeddings import to_graph_tensors

    g = _load_graph(step_file)
    tensors = to_graph_tensors(g)
    result = json.dumps(tensors, indent=2)

    if output:
        Path(output).write_text(result)
        console.print(f"[green]Embeddings written to {output}[/green]")
        console.print(f"  Nodes: {tensors['num_nodes']}, Edges: {tensors['num_edges']}, "
                      f"Feature dim: {tensors['feature_dim']}")
    else:
        click.echo(result)


@cli.command()
@click.argument("step_file")
@click.option("--format", "fmt", type=click.Choice(["json", "table", "yaml"]), default="table")
def metadata(step_file: str, fmt: str):
    """Extract STEP file metadata (header, products, entity stats)."""
    from next3d.core.step_metadata import extract_metadata

    path = Path(step_file)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        sys.exit(1)

    meta = extract_metadata(path)

    if fmt in ("json", "yaml"):
        click.echo(_format_output(meta.to_dict(), fmt))
        return

    console.print(f"\n[bold]STEP Metadata: {step_file}[/bold]\n")

    # Header
    h = meta.header
    ht = Table(title="File Header")
    ht.add_column("Field", style="cyan")
    ht.add_column("Value")
    ht.add_row("Schema", h.schema or "(unknown)")
    ht.add_row("AP Version", h.ap_version or "(unknown)")
    ht.add_row("Originating System", h.originating_system or "(unknown)")
    ht.add_row("Preprocessor", h.preprocessor or "(unknown)")
    ht.add_row("Description", h.description or "(none)")
    ht.add_row("Author", h.author or "(unknown)")
    console.print(ht)

    # Products
    if meta.products:
        pt = Table(title="Products")
        pt.add_column("ID", style="dim")
        pt.add_column("Name", style="yellow")
        pt.add_column("Description")
        for p in meta.products:
            pt.add_row(p.product_id, p.name, p.description)
        console.print(pt)

    # Entity stats
    console.print(f"\nTotal STEP entities: [bold]{meta.total_entities}[/bold]")

    et = Table(title="Top Entity Types")
    et.add_column("Type", style="cyan")
    et.add_column("Count", style="green", justify="right")
    for t, c in sorted(meta.entity_counts.items(), key=lambda x: -x[1])[:15]:
        et.add_row(t, str(c))
    console.print(et)

    # Capabilities
    caps = []
    if meta.has_pmi:
        caps.append("PMI/GD&T")
    if meta.has_colors:
        caps.append("Colors")
    if meta.has_layers:
        caps.append("Layers")
    if meta.has_assembly:
        caps.append("Assembly structure")
    if caps:
        console.print(f"\nDetected capabilities: {', '.join(caps)}")
    else:
        console.print("\n[dim]No extended capabilities (basic B-Rep only)[/dim]")


@cli.command()
@click.argument("step_file")
@click.option("--format", "fmt", type=click.Choice(["json", "table", "yaml"]), default="table")
def mating(step_file: str, fmt: str):
    """Analyze potential mating conditions between features."""
    from next3d.core.assembly import detect_mating_conditions

    g = _load_graph(step_file)
    conditions = detect_mating_conditions(g)

    if fmt in ("json", "yaml"):
        data = [
            {
                "source_id": c.source_id,
                "target_id": c.target_id,
                "mate_type": c.mate_type,
                "parameters": c.parameters,
                "description": c.description,
            }
            for c in conditions
        ]
        click.echo(_format_output(data, fmt))
        return

    if not conditions:
        console.print("[dim]No mating conditions detected.[/dim]")
        return

    console.print(f"\n[bold]Mating Analysis: {step_file}[/bold]\n")

    table = Table(title="Mating Conditions")
    table.add_column("Type", style="yellow")
    table.add_column("Source", style="dim")
    table.add_column("Target", style="dim")
    table.add_column("Description")
    for c in conditions:
        table.add_row(
            c.mate_type,
            c.source_id[:16] + "...",
            c.target_id[:16] + "...",
            c.description,
        )
    console.print(table)

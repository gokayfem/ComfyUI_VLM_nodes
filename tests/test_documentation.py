"""Keep the shipped documentation honest about what the pack actually contains.

The README node reference drifted to 47 undocumented nodes before these checks
existed, and the packaged license metadata disagreed with the LICENSE file.
Both are cheap to assert and expensive to notice by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import ComfyUI_VLM_nodes as package

REPOSITORY = Path(package.__file__).parent


def read(name: str) -> str:
    return (REPOSITORY / name).read_text(encoding="utf-8")


def project_field(name: str) -> str:
    """Read a top-level [project] string field.

    Deliberately regex-based rather than tomllib: this suite also runs on
    Python 3.10, which has no tomllib in the standard library.
    """

    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', read("pyproject.toml"), re.M)
    assert match is not None, f"pyproject.toml has no {name} field."
    return match.group(1)


def test_every_registered_node_appears_in_the_readme():
    readme = read("README.md")
    documented = set(re.findall(r"`([^`]+)`", readme))
    missing = sorted(set(package.NODE_CLASS_MAPPINGS) - documented)
    assert not missing, (
        "These nodes are registered but never named in README.md. "
        f"Add them to the node reference: {missing}"
    )


def test_node_reference_matches_registered_output_types():
    row_pattern = re.compile(
        r"^\|[^|]+\|\s*`(?P<node_id>[^`]+)`\s*\|(?P<outputs>[^|]*)\|$",
        re.M,
    )
    documented = {
        match.group("node_id"): tuple(
            re.findall(r"`([^`]+)`", match.group("outputs"))
        )
        for match in row_pattern.finditer(read("README.md"))
    }
    mismatches = {}
    for node_id, node_class in package.NODE_CLASS_MAPPINGS.items():
        expected = tuple(
            "*" if output is any else str(output)
            for output in node_class.RETURN_TYPES
        )
        if documented.get(node_id) != expected:
            mismatches[node_id] = {
                "documented": documented.get(node_id),
                "registered": expected,
            }

    assert not mismatches, (
        "README.md output schemas do not match the registered RETURN_TYPES: "
        f"{mismatches}"
    )


def test_declared_license_matches_the_license_file():
    declared = project_field("license")
    license_text = read("LICENSE")

    if "Apache License" in license_text:
        expected = "Apache-2.0"
    elif "MIT License" in license_text:
        expected = "MIT"
    else:
        raise AssertionError("Could not identify the license in LICENSE.")

    assert declared == expected, (
        f"pyproject.toml declares {declared!r} but LICENSE is {expected}. "
        "This metadata is embedded in built distribution artifacts."
    )


def test_changelog_documents_the_current_version():
    version = project_field("version")
    changelog = read("CHANGELOG.md")
    assert f"[{version}]" in changelog, (
        f"pyproject version {version} has no CHANGELOG.md entry. The Comfy "
        "Registry only publishes on a version change, so every release needs "
        "one."
    )


def test_contributor_and_security_docs_are_present():
    for name in ("CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "LICENSE"):
        assert (REPOSITORY / name).is_file(), f"{name} is missing."


def test_issue_templates_are_valid_and_request_diagnostics():
    template_dir = REPOSITORY / ".github" / "ISSUE_TEMPLATE"
    bug_report = (template_dir / "bug_report.yml").read_text(encoding="utf-8")
    # Environment detail is what the historically unresolvable reports lacked.
    assert "VLMRuntimeDiagnostics" in bug_report or "Diagnostics" in bug_report
    assert "Node pack version" in bug_report

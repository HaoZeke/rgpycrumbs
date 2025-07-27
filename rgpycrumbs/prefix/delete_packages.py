#!/usr/bin/env python3
# This script follows the guidelines laid out here:
# https://realpython.com/python-script-structure/
#
# Needed since there's no API upstream
# https://github.com/prefix-dev/prefix-dev/issues/29

# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "click==8.1.6",
#   "requests==2.31.0",
#   "rich==13.7.1",
# ]
# ///

import logging
import os
import sys

import click
import requests
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

# --- Setup Logging ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    ],
)
log = logging.getLogger("rich")

# --- Constants ---
BASE_URL = "https://prefix.dev"
PLATFORMS = ["linux-64", "osx-64", "osx-arm64", "win-64", "linux-aarch64", "noarch"]

# --- Core Logic ---


def get_packages_to_delete(channel, package_name):
    """
    Fetches repodata.json for all platforms and finds packages matching the name.
    Returns a list of tuples: (platform, filename).
    """
    packages_to_delete = []
    log.info(f"Fetching package lists for channel [bold cyan]{channel}[/bold cyan]...")

    with Progress() as progress:
        task = progress.add_task("[green]Checking platforms...", total=len(PLATFORMS))
        for platform in PLATFORMS:
            progress.update(
                task, advance=1, description=f"Checking [green]{platform}[/green]"
            )
            repodata_url = f"{BASE_URL}/{channel}/{platform}/repodata.json"
            try:
                response = requests.get(repodata_url)
                if response.status_code == 404:
                    log.debug(f"No repodata.json found for {platform}, skipping.")
                    continue
                response.raise_for_status()
                repodata = response.json()

                # Check both .tar.bz2 and .conda package keys
                all_packages = {
                    **repodata.get("packages", {}),
                    **repodata.get("packages.conda", {}),
                }

                found_count = 0
                import re

                package_pattern = re.compile(rf"^{re.escape(package_name)}-\d+(\.\d+)*")
                for filename in all_packages.keys():
                    if package_pattern.match(filename):
                        packages_to_delete.append((platform, filename))
                        found_count += 1

                if found_count > 0:
                    log.info(
                        f"Found {found_count} package(s) on [cyan]{platform}[/cyan]"
                    )

            except requests.RequestException as e:
                log.error(f"Failed to fetch repodata for {platform}: {e}")

    return packages_to_delete


def delete_package(session, channel, platform, filename, dry_run=False):
    """Sends a DELETE request for a single package file."""
    delete_url = f"{BASE_URL}/api/v1/delete/{channel}/{platform}/{filename}"

    if dry_run:
        log.info(
            f"[DRY RUN] Would delete [yellow]{filename}[/yellow] from [cyan]{platform}[/cyan]"
        )
        return True

    try:
        response = session.delete(delete_url)
        if response.status_code in [200, 204]:
            log.info(
                f"[SUCCESS] Deleted [yellow]{filename}[/yellow] from [cyan]{platform}[/cyan]"
            )
            return True
        else:
            log.error(
                f"[FAILURE] Failed to delete {filename} from {platform}. Status: {response.status_code}, Body: {response.text}"
            )
            return False
    except requests.RequestException as e:
        log.error(f"API request failed for {filename}: {e}")
        return False


# --- Command-Line Interface ---


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--channel",
    required=True,
    help="The name of the prefix.dev channel (e.g., 'rg-forge').",
)
@click.option(
    "--package-name",
    required=True,
    help="The name of the package to delete (e.g., 'eon'). All files starting with this name will be targeted.",
)
@click.option(
    "--api-key",
    envvar="PREFIX_API_KEY",
    help="Your prefix.dev API key. Can also be set via the PREFIX_API_KEY environment variable.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show which packages would be deleted without actually deleting them.",
)
def main(channel, package_name, api_key, dry_run):
    """
    Finds and deletes all versions of a package from a prefix.dev channel.
    """
    if not api_key and not dry_run:
        api_key = click.prompt("Please enter your prefix.dev API key", hide_input=True)
        if not api_key:  # Ensure the API key is set after prompting
            log.error("API key is required for deletion. Exiting.")
            sys.exit(1)
    packages = get_packages_to_delete(channel, package_name)

    if not packages:
        log.info(
            f"No packages matching '{package_name}*' found in channel '{channel}'. Nothing to do."
        )
        return

    log.info(
        f"Found [bold yellow]{len(packages)}[/bold yellow] total package files to delete."
    )
    if not click.confirm("Do you want to proceed?", default=False):
        log.info("Aborted.")
        return

    session = requests.Session()
    if not dry_run and api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})

    success_count = 0
    failure_count = 0

    with Progress() as progress:
        task = progress.add_task("[red]Deleting packages...", total=len(packages))
        for platform, filename in packages:
            if delete_package(session, channel, platform, filename, dry_run):
                success_count += 1
            else:
                failure_count += 1
            progress.update(task, advance=1)

    log.info("-" * 80)
    log.info("Deletion summary:")
    log.info(f"  [green]Packages deleted: {success_count}[/green]")
    log.info(f"  [red]Failures: {failure_count}[/red]")


if __name__ == "__main__":
    main()

"""
Git state logging for reproducibility.

Logs the current git branch, commit hash, and any uncommitted changes
to enable full reproducibility of experiment runs.
"""

import logging
import os
from typing import Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def log_git_state(
    strict: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log current git state for reproducibility.

    Logs the current branch and commit hash. If there are uncommitted
    changes to tracked files, logs the full diff (for reproducibility)
    or raises an exception if strict=True.

    Untracked files are ignored - if they're relevant for reproducibility,
    they should be staged with `git add`.

    :param strict: If True, raise exception on uncommitted changes.
        If False (default), log the diff as a warning.
    :param logger: Logger to use. If None, uses module logger.

    :raises RuntimeError: If strict=True and there are uncommitted changes.
    """
    if logger is None:
        logger = module_logger

    # Suppress GitPython's verbose DEBUG logging before import
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)

    try:
        from git import Repo, InvalidGitRepositoryError
    except ImportError:
        logger.error(
            "GitPython not installed. Cannot log git state for reproducibility. "
            "Install with: pip install gitpython"
        )
        return

    try:
        repo = Repo(search_parent_directories=True)
    except InvalidGitRepositoryError:
        logger.warning("Not in a git repository. Cannot log git state.")
        return

    # Get repo name from working directory
    repo_name = os.path.basename(repo.working_dir)

    # Get branch name
    try:
        branch = repo.active_branch.name
    except TypeError:
        # Detached HEAD state
        branch = "DETACHED HEAD"

    # Get commit info
    commit = repo.head.commit.hexsha
    commit_short = commit[:8]
    commit_message = repo.head.commit.message.strip()

    logger.info(f"Git state: {repo_name} @ {branch} @ {commit_short}")
    logger.info(f"Commit message:\n{commit_message}")

    # Check for uncommitted changes (tracked files only)
    is_dirty = repo.is_dirty(untracked_files=False)

    if is_dirty:
        # Get the diff for tracked files
        diff_output = repo.git.diff()
        staged_diff = repo.git.diff("--cached")

        if strict:
            msg = (
                "Uncommitted changes detected and strict=True. "
                "Please commit your changes before running experiments."
            )
            if diff_output:
                msg += f"\n\nUnstaged changes:\n{diff_output}"
            if staged_diff:
                msg += f"\n\nStaged changes:\n{staged_diff}"
            raise RuntimeError(msg)

        # Log warning with full diff for reproducibility
        logger.warning("Uncommitted changes detected. Logging diff for reproducibility:")

        if diff_output:
            logger.warning(f"Unstaged changes:\n{diff_output}\n")

        if staged_diff:
            logger.warning(f"Staged changes:\n{staged_diff}\n")
    else:
        logger.info("  Working tree clean")

    # Check for untracked files and provide guidance
    untracked = repo.untracked_files
    if untracked:
        logger.warning(
            f"{len(untracked)} untracked file(s). If relevant for reproducibility, "
            "use `git add <files>` to start tracking them."
        )

    logger.info("=" * 60 + "\n\n")

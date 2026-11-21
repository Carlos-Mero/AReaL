# SPDX-License-Identifier: Apache-2.0

"""AReaL Sync RPC Server — Guard + Data + Engine composition.

This module composes the shared Guard with data and engine blueprints
to create the full RPC server used by training workers.

Usage::

    python -m areal.infra.rpc.rpc_server \\
        --experiment-name exp1 --trial-name trial1 \\
        --role actor --worker-index 0
"""

from __future__ import annotations

import logging as stdlib_logging
import resource

from areal.infra.rpc.guard.app import (
    GuardState,
    configure_state_from_args,
    create_app,
    make_base_parser,
    run_server,
)
from areal.infra.rpc.guard.data_blueprint import data_bp
from areal.infra.rpc.guard.engine_blueprint import engine_bp, register_engine_hooks
from areal.infra.utils.http import HTTPX_MAX_CONNECTIONS
from areal.utils import logging, perf_tracer

logger = logging.getLogger("SyncRPCServer")


def _configure_nofile_limit() -> None:
    """Budget descriptors for incoming RPCs, proxy connections and engine I/O."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 4 * HTTPX_MAX_CONNECTIONS
    if soft == resource.RLIM_INFINITY or soft >= target:
        return
    limit = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if limit > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
        except (OSError, ValueError) as exc:
            logger.warning("Could not raise the open-file limit from %d: %s", soft, exc)
            return
    if limit < target:
        logger.warning(
            "Worker open-file limit is capped at %d. High rollout concurrency "
            "may require a higher hard limit.",
            limit,
        )


def main():
    parser = make_base_parser(
        description="AReaL Sync RPC Server for TrainEngine/InferenceEngine"
    )
    parser.add_argument(
        "--werkzeug-log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for Werkzeug (Flask's WSGI server). Default: WARNING",
    )

    args, _ = parser.parse_known_args()
    _configure_nofile_limit()

    werkzeug_logger = stdlib_logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(getattr(stdlib_logging, args.werkzeug_log_level))

    state = GuardState()
    bind_host = configure_state_from_args(state, args)

    app = create_app(state)
    app.register_blueprint(data_bp)
    app.register_blueprint(engine_bp)
    register_engine_hooks(state)

    state.register_cleanup_hook(lambda: perf_tracer.save(force=True))

    logger.info(f"Werkzeug log level: {args.werkzeug_log_level}")

    run_server(state, app, bind_host, args.port)


if __name__ == "__main__":
    main()

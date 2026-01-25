import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.modules.data.methods import get_dataset_data, get_datasets, get_datasets_by_name
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.operations import (
    log_pipeline_run_complete,
    log_pipeline_run_error,
    log_pipeline_run_start,
)
from cognee.modules.pipelines.operations.run_tasks_data_item import run_tasks_data_item
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.users.methods import get_default_user


def _extract_item_status(item):
    if isinstance(item, dict):
        run_info = item.get("run_info")
        data_id = item.get("data_id")
    else:
        run_info = item
        data_id = None

    if run_info is None:
        return data_id, "Unknown", None

    if hasattr(run_info, "status"):
        status = run_info.status
        payload = getattr(run_info, "payload", None)
    elif isinstance(run_info, dict):
        status = run_info.get("status")
        payload = run_info.get("payload")
    else:
        status = str(run_info)
        payload = None

    return data_id, status, payload


def _resolve_log_path(log_file):
    if log_file:
        return Path(log_file)

    return Path("/") / "cognify_batch.log"


def _get_logger(log_file):
    logger = logging.getLogger("cognify_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = None
    try:
        log_path = _resolve_log_path(log_file)
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except Exception:
        handler = logging.FileHandler("cognify_batch.log", encoding="utf-8")
        log_path = Path("cognify_batch.log").resolve()

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, log_path


def _sort_data_items(data_items):
    return sorted(
        data_items,
        key=lambda item: (item.created_at or datetime.min, str(item.id)),
    )


def _is_unprocessed(data_item, dataset_id):
    pipeline_status = data_item.pipeline_status or {}
    dataset_status = pipeline_status.get("cognify_pipeline", {}).get(str(dataset_id))
    return dataset_status != DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED


async def _report_per_item_status(datasets, cognify_result):
    user = await get_default_user()
    if datasets is None:
        dataset_list = await get_datasets(user.id)
    else:
        dataset_list = await get_datasets_by_name(datasets, user.id)

    for dataset in dataset_list:
        run_info = cognify_result.get(str(dataset.id)) or cognify_result.get(dataset.id)
        if not run_info:
            continue

        data_items = _sort_data_items(await get_dataset_data(dataset.id))
        data_index = {str(data.id): idx for idx, data in enumerate(data_items)}
        data_lookup = {str(data.id): data for data in data_items}

        ingestion_info = getattr(run_info, "data_ingestion_info", None)
        if not ingestion_info:
            print(f"{dataset.name}: no per-item status available (background run or empty dataset).")
            continue

        print(f"{dataset.name}:")
        for item in ingestion_info:
            data_id, status, payload = _extract_item_status(item)
            data_id_str = str(data_id) if data_id else None
            data = data_lookup.get(data_id_str) if data_id_str else None
            index = data_index.get(data_id_str) if data_id_str else None
            index_label = str(index + 1) if index is not None else "?"
            name = data.name if data and data.name else data_id_str or "unknown"
            line = f"  [{index_label}] {name}: {status}"
            if status == "PipelineRunErrored" and payload:
                line = f"{line} ({payload})"
            print(line)


async def _get_datasets(datasets):
    user = await get_default_user()
    if datasets is None:
        return await get_datasets(user.id)
    return await get_datasets_by_name(datasets, user.id)


async def _run_batch(dataset, batch_size, logger, fail_on_index):
    user = await get_default_user()
    data_items = _sort_data_items(await get_dataset_data(dataset.id))
    unprocessed = [item for item in data_items if _is_unprocessed(item, dataset.id)]
    batch = unprocessed[:batch_size]

    if not batch:
        logger.info("Dataset %s: no unprocessed items found.", dataset.name)
        return None

    logger.info(
        "Dataset %s: processing %s/%s unprocessed items.",
        dataset.name,
        len(batch),
        len(unprocessed),
    )

    fail_after = False
    injected_error = None
    bad_item = None
    bad_item_label = "unknown"
    if fail_on_index is not None:
        if fail_on_index < 1 or fail_on_index > len(batch):
            logger.info(
                "Fail index %s is out of range for batch size %s; no failure injected.",
                fail_on_index,
                len(batch),
            )
        else:
            bad_item = batch[fail_on_index - 1]
            batch = batch[: fail_on_index - 1]
            fail_after = True
            bad_item_label = (
                bad_item.name if bad_item and bad_item.name else str(bad_item.id) if bad_item else "unknown"
            )
            logger.info(
                "Injecting failure after processing %s item(s). Failing item %s (%s).",
                fail_on_index - 1,
                fail_on_index,
                bad_item_label,
            )

    tasks = await get_default_tasks(user=user)
    pipeline_name = "cognify_pipeline"
    pipeline_id = str(generate_pipeline_id(user.id, dataset.id, pipeline_name))
    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset.id, batch)
    pipeline_run_id = pipeline_run.pipeline_run_id

    data_index = {str(data.id): idx for idx, data in enumerate(data_items)}
    any_error = False

    for idx, data_item in enumerate(batch, start=1):
        if fail_after and idx == fail_on_index:
            injected_error = RuntimeError(
                f"Injected failure for batch item {fail_on_index}: {bad_item_label}"
            )
            logger.error("%s", injected_error)
            any_error = True
            break

        logger.info(
            "Dataset %s: processing item %s (%s).",
            dataset.name,
            idx,
            data_item.name or data_item.id,
        )
        try:
            result = await run_tasks_data_item(
                data_item=data_item,
                dataset=dataset,
                tasks=tasks,
                pipeline_name=pipeline_name,
                pipeline_id=pipeline_id,
                pipeline_run_id=pipeline_run_id,
                context={"dataset": dataset},
                user=user,
                incremental_loading=True,
            )
            data_id, status, payload = _extract_item_status(result)
            data_id_str = str(data_id) if data_id else None
            index = data_index.get(data_id_str) if data_id_str else None
            index_label = str(index + 1) if index is not None else "?"
            name = data_item.name if data_item.name else data_id_str or "unknown"
            message = f"[{index_label}] {name}: {status}"
            if status == "PipelineRunErrored" and payload:
                message = f"{message} ({payload})"
                any_error = True
            logger.info("Dataset %s: %s", dataset.name, message)
        except Exception as exc:
            any_error = True
            logger.exception(
                "Dataset %s: item %s (%s) failed: %s",
                dataset.name,
                idx,
                data_item.name or data_item.id,
                exc,
            )

    if any_error:
        await log_pipeline_run_error(
            pipeline_run_id,
            pipeline_id,
            pipeline_name,
            dataset.id,
            batch,
            injected_error or RuntimeError("Batch completed with errors"),
        )
    else:
        await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset.id, batch
        )

    if fail_after:
        if injected_error is None:
            injected_error = RuntimeError("Injected failure")
        raise injected_error

    return None


async def run_cognify(datasets, batch_size, report_items, log_file, fail_on_index):
    logger, log_path = _get_logger(log_file)
    print(f"Logging to: {log_path}")
    logger.info("Starting cognify batch run.")

    dataset_list = await _get_datasets(datasets)
    results = {}

    for dataset in dataset_list:
        logger.info("Running batch for dataset: %s (%s)", dataset.name, dataset.id)
        try:
            run_info = await _run_batch(dataset, batch_size, logger, fail_on_index)
            if run_info:
                results[str(dataset.id)] = run_info
            else:
                results[str(dataset.id)] = None
        except Exception as exc:
            logger.exception("Batch failed for dataset %s: %s", dataset.name, exc)
            results[str(dataset.id)] = None

    print(results)
    if report_items:
        await _report_per_item_status(datasets, results)

    # Always log per-item status when available
    if results:
        await _log_per_item_status(datasets, results, logger)


async def _log_per_item_status(datasets, cognify_result, logger):
    user = await get_default_user()
    if datasets is None:
        dataset_list = await get_datasets(user.id)
    else:
        dataset_list = await get_datasets_by_name(datasets, user.id)

    for dataset in dataset_list:
        run_info = cognify_result.get(str(dataset.id)) or cognify_result.get(dataset.id)
        if not run_info:
            logger.info("Dataset %s: no run info available.", dataset.name)
            continue

        data_items = _sort_data_items(await get_dataset_data(dataset.id))
        data_index = {str(data.id): idx for idx, data in enumerate(data_items)}
        data_lookup = {str(data.id): data for data in data_items}

        ingestion_info = getattr(run_info, "data_ingestion_info", None)
        if not ingestion_info:
            logger.info("Dataset %s: no per-item status available.", dataset.name)
            continue

        for item in ingestion_info:
            data_id, status, payload = _extract_item_status(item)
            data_id_str = str(data_id) if data_id else None
            data = data_lookup.get(data_id_str) if data_id_str else None
            index = data_index.get(data_id_str) if data_id_str else None
            index_label = str(index + 1) if index is not None else "?"
            name = data.name if data and data.name else data_id_str or "unknown"
            message = f"[{index_label}] {name}: {status}"
            if status == "PipelineRunErrored" and payload:
                message = f"{message} ({payload})"
            logger.info("Dataset %s: %s", dataset.name, message)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cognify in batches of data items (e.g. 10 documents at a time)."
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=["main_dataset"],
        help="Dataset name(s) to cognify. Default: main_dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Cognify all datasets for the current user.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of data items to process per batch. Default: 10",
    )
    parser.add_argument(
        "--report-items",
        action="store_true",
        help="Print per-document status from the cognify run.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path. Default: /cognify_batch.log (falls back to ./cognify_batch.log).",
    )
    parser.add_argument(
        "--fail-on-index",
        type=int,
        default=None,
        help="Inject a failure before processing the Nth item in the batch (1-based).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = None if args.all else args.dataset
    asyncio.run(
        run_cognify(datasets, args.batch_size, args.report_items, args.log_file, args.fail_on_index)
    )


if __name__ == "__main__":
    main()

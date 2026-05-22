#!/usr/bin/env python3
"""国会会議録検索システムAPIからレコードを収集するCLI。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE_URL = "https://kokkai.ndl.go.jp/api"
API_RECORD_LIMITS = {
    "meeting": 10,
    "meeting_list": 100,
    "speech": 100,
}
API_PATHS = {
    "meeting": "meeting",
    "meeting_list": "meeting_list",
    "speech": "speech",
}
API_RECORD_KEYS = {
    "meeting": "meetingRecord",
    "meeting_list": "meetingRecord",
    "speech": "speechRecord",
}
SEARCH_ARG_TO_API_PARAM = {
    "any": "any",
    "name_of_house": "nameOfHouse",
    "name_of_meeting": "nameOfMeeting",
    "speaker": "speaker",
    "from_date": "from",
    "until_date": "until",
    "speech_id": "speechID",
    "issue_id": "issueID",
    "session_from": "sessionFrom",
    "session_to": "sessionTo",
    "speaker_position": "speakerPosition",
    "speaker_group": "speakerGroup",
    "speaker_role": "speakerRole",
    "speech": "speech",
}


class KokkaiAPIError(RuntimeError):
    """APIリクエストまたはレスポンス処理に失敗した。"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("0以上の整数を指定してください")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("0以上の数値を指定してください")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="国会会議録検索システムAPIの取得結果をJSONLに保存します。"
    )
    parser.add_argument(
        "--endpoint",
        choices=tuple(API_PATHS),
        default="meeting",
        help="取得するAPI。meetingは会議録本文、speechは発言単位、meeting_listは会議一覧です。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/kokkai/records.jsonl"),
        help="取得レコードを1行1件で保存するJSONLファイル。",
    )
    parser.add_argument(
        "--pages-output-dir",
        type=Path,
        help="APIレスポンスをページごとに残すディレクトリ。",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="出力JSONLへ追記します。既定では上書きします。",
    )
    parser.add_argument(
        "--start-record",
        type=positive_int,
        default=1,
        help="取得開始位置。中断後の再開時にAPIのnextRecordPositionを指定できます。",
    )
    parser.add_argument(
        "--maximum-records",
        type=positive_int,
        help="1リクエストあたりの取得件数。API上限を超える場合は上限に丸めます。",
    )
    parser.add_argument(
        "--max-pages",
        type=positive_int,
        help="取得ページ数の上限。試し取りでは1などを指定してください。",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=non_negative_float,
        default=3.0,
        help="連続リクエスト間の待機秒数。API案内に合わせて既定は3秒です。",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_int,
        default=30,
        help="HTTPタイムアウト秒数。",
    )
    parser.add_argument(
        "--retries",
        type=non_negative_int,
        default=2,
        help="一時的なHTTP失敗時の再試行回数。",
    )
    parser.add_argument("--any", help="発言本文などを横断する検索語。")
    parser.add_argument("--name-of-house", help="院名。例: 衆議院")
    parser.add_argument("--name-of-meeting", help="会議名。例: 予算委員会")
    parser.add_argument("--speaker", help="発言者名。")
    parser.add_argument("--from-date", help="開始日。YYYY-MM-DD形式。")
    parser.add_argument("--until-date", help="終了日。YYYY-MM-DD形式。")
    parser.add_argument("--speech-id", help="発言ID。")
    parser.add_argument("--issue-id", help="会議録ID。")
    parser.add_argument("--session-from", type=positive_int, help="開始回次。")
    parser.add_argument("--session-to", type=positive_int, help="終了回次。")
    parser.add_argument("--speaker-position", help="肩書き。")
    parser.add_argument("--speaker-group", help="会派名。")
    parser.add_argument("--speaker-role", help="発言者役割。")
    parser.add_argument(
        "--speech",
        help="発言本文検索語。APIのspeech検索条件を使います。",
    )
    return parser.parse_args()


def selected_search_params(args: argparse.Namespace) -> dict[str, str | int]:
    params: dict[str, str | int] = {}
    for arg_name, api_name in SEARCH_ARG_TO_API_PARAM.items():
        value = getattr(args, arg_name)
        if value is not None:
            params[api_name] = value
    return params


def capped_maximum_records(endpoint: str, requested: int | None) -> int:
    limit = API_RECORD_LIMITS[endpoint]
    if requested is None:
        return limit
    return min(requested, limit)


def build_request_url(endpoint: str, params: dict[str, str | int]) -> str:
    query = urlencode(params)
    return f"{API_BASE_URL}/{API_PATHS[endpoint]}?{query}"


def fetch_page(
    endpoint: str,
    params: dict[str, str | int],
    timeout_seconds: int,
    retries: int,
) -> dict[str, Any]:
    url = build_request_url(endpoint, params)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "meeting-score-kokkai-collector/1.0",
        },
    )

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.load(response)
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if retryable and attempt < retries:
                wait_before_retry(attempt)
                continue
            raise KokkaiAPIError(f"HTTP {exc.code}: {url}") from exc
        except (TimeoutError, URLError) as exc:
            if attempt < retries:
                wait_before_retry(attempt)
                continue
            raise KokkaiAPIError(f"APIに接続できませんでした: {url}") from exc
        except json.JSONDecodeError as exc:
            raise KokkaiAPIError(f"JSONレスポンスを解釈できませんでした: {url}") from exc

        if not isinstance(payload, dict):
            raise KokkaiAPIError(f"想定外のJSONレスポンスです: {url}")
        return payload

    raise AssertionError("retry loop must return or raise")


def wait_before_retry(attempt: int) -> None:
    time.sleep(2**attempt)


def payload_records(endpoint: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    api_error = payload.get("message")
    if isinstance(api_error, str):
        details = payload.get("details")
        if isinstance(details, list) and details:
            detail_text = " / ".join(str(detail) for detail in details)
            raise KokkaiAPIError(f"APIエラー: {api_error}: {detail_text}")
        raise KokkaiAPIError(f"APIエラー: {api_error}")

    record_key = API_RECORD_KEYS[endpoint]
    records = payload.get(record_key)
    if records is None:
        raise KokkaiAPIError(f"{record_key}がレスポンスにありません")
    if not isinstance(records, list):
        raise KokkaiAPIError(f"{record_key}が配列ではありません")
    if any(not isinstance(record, dict) for record in records):
        raise KokkaiAPIError(f"{record_key}にオブジェクト以外の値が含まれています")
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], append: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False))
            output.write("\n")
            count += 1
    return count


def write_page(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2))
            output.write("\n")
    except FileExistsError as exc:
        raise KokkaiAPIError(f"ページJSONが既に存在します: {path}") from exc


def collect(args: argparse.Namespace) -> int:
    search_params = selected_search_params(args)
    if not search_params:
        raise KokkaiAPIError(
            "検索条件がありません。例: --from-date 2024-01-01 --until-date 2024-01-31"
        )

    maximum_records = capped_maximum_records(args.endpoint, args.maximum_records)
    next_record_position: int | None = args.start_record
    page_number = 0
    total_written = 0
    first_write = not args.append

    while next_record_position is not None:
        page_number += 1
        params = {
            **search_params,
            "recordPacking": "json",
            "startRecord": next_record_position,
            "maximumRecords": maximum_records,
        }
        payload = fetch_page(
            args.endpoint,
            params,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )
        records = payload_records(args.endpoint, payload)
        total_written += write_jsonl(args.output, records, append=not first_write)
        first_write = False

        if args.pages_output_dir is not None:
            page_path = args.pages_output_dir / f"start_record_{next_record_position:09d}.json"
            write_page(page_path, payload)

        print(
            f"page={page_number} records={len(records)} total_written={total_written} "
            f"next={payload.get('nextRecordPosition')}",
            file=sys.stderr,
        )

        if args.max_pages is not None and page_number >= args.max_pages:
            break

        next_record_position = payload.get("nextRecordPosition")
        if next_record_position is None:
            break
        if not isinstance(next_record_position, int):
            raise KokkaiAPIError("nextRecordPositionが整数ではありません")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return total_written


def main() -> int:
    args = parse_args()
    try:
        total_written = collect(args)
    except KokkaiAPIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"saved {total_written} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

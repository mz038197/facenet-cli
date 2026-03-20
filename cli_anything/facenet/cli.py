import json
import os

import click

from cli_anything.facenet.core.recognition import (
    compare_images,
    capture_embedding_once,
    recognize_folder_images_to_csv,
    recognize_match_name,
)


def emit(payload, as_json: bool):
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        click.echo(payload)


@click.group()
@click.option("--json", "as_json", is_flag=True, help="輸出 JSON 格式")
@click.option("--workspace", default=".", help="facenet-pytorch 專案目錄")
@click.pass_context
def cli(ctx: click.Context, as_json: bool, workspace: str):
    ctx.ensure_object(dict)
    ctx.obj["json"] = as_json
    ctx.obj["workspace"] = os.path.abspath(workspace)


@cli.group("embedding")
def embedding():
    """拍照/影像辨識命令"""


@embedding.command("help")
@click.pass_context
def embedding_help(ctx: click.Context):
    commands = [
        {
            "command": "capture",
            "description": "拍照擷取 embedding，可選輸出 CSV",
            "example": "embedding capture --output-csv out.csv",
        },
        {
            "command": "find",
            "description": "拍照後與資料庫比對，輸出姓名與距離",
            "example": "embedding find --database-csv face_embeddings_database.csv --threshold 1.0",
        },
        {
            "command": "images",
            "description": "批次讀取資料夾影像，輸出 filename + embedding CSV",
            "example": "embedding images --input-folder face_images --output-csv batch.csv",
        },
        {
            "command": "compare",
            "description": "比對兩張影像，輸出距離與是否相同",
            "example": "embedding compare --image1 a.jpg --image2 b.jpg --threshold 1.2",
        },
    ]
    payload = {
        "ok": True,
        "group": "embedding",
        "commands": commands,
    }
    emit(payload, ctx.obj["json"])


@embedding.command("capture")
@click.option("--output-csv", default=None, help="可選：將單次 embedding 寫入 CSV")
@click.option("--dry-run", is_flag=True, help="只檢查流程，不啟動攝影機")
@click.pass_context
def embedding_capture(ctx: click.Context, output_csv: str | None, dry_run: bool):
    result = capture_embedding_once(
        workspace=ctx.obj["workspace"],
        output_csv=output_csv,
        dry_run=dry_run,
    )
    emit(result, ctx.obj["json"])


@embedding.command("find")
@click.option("--database-csv", default="face_embeddings_database.csv", help="人臉 embeddings 資料庫 CSV")
@click.option("--threshold", default=1.0, type=float, help="辨識距離閾值")
@click.option("--dry-run", is_flag=True, help="只檢查流程，不啟動攝影機")
@click.pass_context
def embedding_match_name(
    ctx: click.Context,
    database_csv: str,
    threshold: float,
    dry_run: bool,
):
    result = recognize_match_name(
        workspace=ctx.obj["workspace"],
        database_csv=database_csv,
        threshold=threshold,
        dry_run=dry_run,
    )
    emit(result, ctx.obj["json"])


@embedding.command("images")
@click.option("--input-folder", required=True, help="要讀取的影像資料夾路徑")
@click.option("--output-csv", required=True, help="辨識結果輸出 CSV 路徑")
@click.option("--dry-run", is_flag=True, help="只檢查流程，不實際處理影像")
@click.pass_context
def embedding_images(
    ctx: click.Context,
    input_folder: str,
    output_csv: str,
    dry_run: bool,
):
    result = recognize_folder_images_to_csv(
        workspace=ctx.obj["workspace"],
        input_folder=input_folder,
        output_csv=output_csv,
        dry_run=dry_run,
    )
    emit(result, ctx.obj["json"])


@embedding.command("compare")
@click.option("--image1", required=True, help="影像位置1")
@click.option("--image2", required=True, help="影像位置2")
@click.option("--threshold", required=True, type=float, help="判定相同人的閾值")
@click.option("--dry-run", is_flag=True, help="只檢查流程，不實際讀圖")
@click.pass_context
def embedding_compare_images(
    ctx: click.Context,
    image1: str,
    image2: str,
    threshold: float,
    dry_run: bool,
):
    result = compare_images(
        image_path_1=image1,
        image_path_2=image2,
        threshold=threshold,
        dry_run=dry_run,
    )
    emit(result, ctx.obj["json"])


def main():
    cli()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils.count import count_objects
from utils.detect import create_fruit_mask
from utils.preprocess import preprocess_image


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "images" / "sample.jpg"
DEFAULT_OUTPUT = BASE_DIR / "output" / "result.jpg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dem so luong trai cay tu anh trong nong nghiep."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Duong dan anh dau vao.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Duong dan anh ket qua.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=700,
        help="Dien tich toi thieu cua mot doi tuong de duoc tinh la trai cay.",
    )
    parser.add_argument(
        "--distance-ratio",
        type=float,
        default=0.35,
        help="Ty le nguong distance transform de tach cac trai cay dang dinh nhau.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hien thi cua so debug khi chay tren may co giao dien.",
    )
    return parser


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return BASE_DIR / path


def main() -> int:
    args = build_parser().parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    import numpy as np

    image = cv2.imdecode(np.fromfile(str(input_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Khong the doc anh dau vao: {input_path}. "
            "Kiem tra lai duong dan hoac noi dung file."
        )

    processed = preprocess_image(image)
    mask = create_fruit_mask(processed)
    count, result, markers = count_objects(
        mask=mask,
        original=image,
        min_area=args.min_area,
        distance_ratio=args.distance_ratio,
    )

    cv2.putText(
        result,
        f"So luong: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    print(f"So luong trai cay: {count}")
    print(f"Da luu ket qua tai: {output_path}")

    if args.show:
        cv2.imshow("Processed", processed)
        cv2.imshow("Mask", mask)
        cv2.imshow("Watershed markers", markers)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

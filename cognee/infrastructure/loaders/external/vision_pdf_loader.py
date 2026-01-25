import asyncio
import os
import tempfile
from typing import List

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class VisionPdfLoader(LoaderInterface):
    """
    PDF loader that renders each page to an image and transcribes it via LLM vision.

    The transcription prompt aims to preserve original content and language as closely
    as possible. Pages are processed in batches to improve throughput.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]

    @property
    def loader_name(self) -> str:
        return "vision_pdf_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """Check if file can be handled by this loader."""
        # Check file extension
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True

        return False

    async def load(self, file_path: str, render_scale: float = 1.5, batch_size: int = 5, **kwargs) -> str:
        """
        Load PDF file by rendering pages to images and transcribing via LLM.

        Args:
            file_path: Path to the PDF file.
            render_scale: Scale factor for rendering; higher gives sharper text.
            batch_size: Number of pages to transcribe concurrently.
            **kwargs: Additional arguments (unused).

        Returns:
            Path to stored text file containing concatenated page transcriptions.

        Raises:
            ImportError: If pypdfium2 is not installed.
            Exception: If PDF processing fails.
        """
        pdf = None
        try:
            import pypdfium2 as pdfium
        except ImportError as e:
            raise ImportError(
                "pypdfium2 is required for PDF vision transcription. Install with: pip install pypdfium2"
            ) from e

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        temp_files: list[str] = []
        content_parts: list[str] = []

        try:
            with open(file_path, "rb") as file:
                file_metadata = await get_file_metadata(file)
            storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

            logger.info(f"Rendering PDF pages for vision transcription: {file_path}")
            pdf = pdfium.PdfDocument(file_path)
            page_indices = list(range(len(pdf)))

            async def transcribe_page(idx: int) -> tuple[int, str]:
                """Render a single page and transcribe it via LLM vision."""
                page = pdf[idx]
                pil_image = page.render(scale=render_scale).to_pil().convert("RGB")

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    pil_image.save(tmp.name, format="JPEG", quality=95, subsampling=0)
                    temp_files.append(tmp.name)
                    image_path = tmp.name

                # LLM vision transcription aims to mirror original page content
                result = await LLMGateway.transcribe_image(image_path)
                page_text = result.choices[0].message.content
                return idx + 1, page_text

            # Batch pages for concurrent transcription
            for start in range(0, len(page_indices), batch_size):
                batch = page_indices[start : start + batch_size]
                batch_results = await asyncio.gather(*(transcribe_page(idx) for idx in batch))
                for page_num, page_text in sorted(batch_results, key=lambda t: t[0]):
                    if page_text and page_text.strip():
                        content_parts.append(f"Page {page_num}:\n{page_text}\n")

            full_content = "\n".join(content_parts)

            storage_config = get_storage_config()
            data_root_directory = storage_config["data_root_directory"]
            storage = get_file_storage(data_root_directory)

            full_file_path = await storage.store(storage_file_name, full_content)
            return full_file_path

        except Exception as e:
            logger.error(f"Failed to process PDF via vision pipeline {file_path}: {e}")
            raise Exception(f"PDF vision processing failed: {e}") from e
        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass
            for path in temp_files:
                try:
                    os.remove(path)
                except OSError:
                    pass

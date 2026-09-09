# Complete runtime for the document-processing skills in .claude/skills:
# docx, pdf, pptx, and xlsx.
FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

# General tooling, DOCX/PPTX/XLSX rendering and validation, PDF tooling,
# OCR, and optional PDF/image workflows documented by the skills.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git tree cloc curl wget bash coreutils findutils grep ca-certificates \
        build-essential gnupg \
        ## skills
        libreoffice pandoc poppler-utils zip unzip \
        qpdf pdftk-java tesseract-ocr imagemagick && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies used by the skills and their bundled scripts.
# MarkItDown's Office extras enable PPTX conversion support.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pytest ruff mypy \
        ## skills
        defusedxml lxml \
        pypdf pdfplumber pandas reportlab \
        pillow pdf2image pytesseract pypdfium2 \
        openpyxl 'markitdown[pptx]'

# uv (fast Python package manager). Installed into /usr/local/bin rather than
# the installer's default /root/.local/bin: Debian's /root is mode 0700, so a
# container run under --user <non-root> cannot reach anything below it however
# the PATH is set.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Node.js 18 and globally available Node dependencies used by DOCX, PPTX,
# and PDF skills. NODE_PATH lets require() resolve them from any workdir.
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm config set prefix /opt/npm-global && \
    npm install -g npm npx typescript webpack webpack-cli \
        ## skills
        docx pptxgenjs pdf-lib react react-dom react-icons sharp && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV NODE_PATH=/opt/npm-global/lib/node_modules
ENV PATH=/opt/npm-global/bin:${PATH}

WORKDIR /workspace

CMD ["sleep", "infinity"]
FROM python:3.12.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ["pyproject.toml", "uv.lock", "./"]

RUN uv sync --frozen --no-install-project --no-dev

COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["uv", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
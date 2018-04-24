# Document Scanner

The goal of this project is to allow me to easily take pictures of documents (mostly letters and invoices) and have them converted to PDFs that are searchable.

## Quickstart

Start by building the Docker image

```
docker build -t docscan .
```

Then given you have a picture `pic.jpg` you can run

```
docker run --rm -v (pwd):/code -w /code docscan:latest python scan.py pic.jpg
```

Use `--help` for more insight into the available options.

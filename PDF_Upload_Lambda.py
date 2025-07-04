import asyncio
import aioboto3
import boto3
import base64
import json
import logging
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase, PDFUrlReader
from agno.vectordb.pgvector import PgVector, SearchType
from requests_toolbelt.multipart import decoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

def lambda_handler(event, context):
    s3_bucket_name = "tl-course-materials"
    s3_bucket_base_url = "https://tl-course-materials.s3.eu-central-1.amazonaws.com"

    knowledge_db_url = "postgresql+psycopg2://postgres.jlqpiruljvmkvumnbtqd:changeit@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"

    try:
        files = extract_files(event['headers'], event['body'], event.get('isBase64Encoded', False))
        asyncio.run(main(s3_bucket_base_url, s3_bucket_name, knowledge_db_url, files))

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            },
            'body': json.dumps('Dateien erfolgreich gespeichert.')
        }
    except Exception as e:
        logger.error("Fehler beim Verarbeiten des Uploads.", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            },
            'body': json.dumps(f'Fehler: {str(e)}')
        }

def extract_files(headers, body, isBase64Encoded):
    files = []

    # Parse Content-Type + Body
    content_type = headers.get('content-type') or headers.get('Content-Type')
    body = base64.b64decode(body) if isBase64Encoded else body

    multipart_data = decoder.MultipartDecoder(body, content_type)
    for part in multipart_data.parts:
        content_disposition = part.headers.get(b'Content-Disposition', b'').decode()

        if 'filename=' in content_disposition:
            # Extract filename + file content
            filename = content_disposition.split('filename="')[1].split('"')[0]
            file_content = part.content

            files.append([ filename, file_content ])

    logger.info("Files extracted.")
    return files

async def main(s3_bucket_base_url, s3_bucket_name, knowledge_db_url, files):
    await save_files_to_s3(s3_bucket_name, files)
    await load_files_from_s3_to_knowledge_base(s3_bucket_base_url, knowledge_db_url, [filename for filename, _ in files])

async def save_files_to_s3(bucket_name, files):
    logger.info("Saving file(s) to S3...")

    session = aioboto3.Session()
    async with session.client("s3") as s3:
        # Start all uploads in parallel
        tasks = [
            s3.put_object(Bucket=bucket_name, Key=filename, Body=file_content, ContentType='application/pdf')
            for filename, file_content in files
        ]

        # Wait for all uploads to be completed
        await asyncio.gather(*tasks)

    logger.info("File(s) successfully saved to S3.")

async def load_files_from_s3_to_knowledge_base(s3_bucket_base_url, knowledge_db_url, filenames):
    logger.info("Loading file(s) from S3 to PDF knowledge base...")

    s3_file_urls = [f"{s3_bucket_base_url}/{filename}" for filename in filenames]
    logger.info("S3 file URLs: %s", s3_file_urls)

    knowledge_base = PDFUrlKnowledgeBase(
        urls=s3_file_urls,
        vector_db=PgVector(
            table_name="pdf_documents", # Table name: ai.pdf_documents
            db_url=knowledge_db_url,
            search_type=SearchType.hybrid,
            # TODO: Parameter wie Embedder, Reranker etc. wichtig?
        ),
        reader = PDFUrlReader(chunk=True)
    )

    await knowledge_base.aload(recreate = not knowledge_base.exists(), upsert=True) # TODO: Metadaten mitgeben? -> z. B. metadata={"user_id": "jordan_mitchell", "document_type": "cv", "year": 2025}

    logger.info("File(s) successfully loaded from S3 to PDF knowledge base.")
# project_core/core/retrieval.py

import logging
from langchain_community.vectorstores.chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from .config import (
    get_embeddings_model,
    VECTORSTORE_PATH,
    DOCSTORE_PATH,
    COLLECTION_NAME
)

# إعداد المسجل
logger = logging.getLogger(__name__)

def create_retriever():
    """
    يقوم بتهيئة وإرجاع المسترجع متعدد المتجهات (MultiVectorRetriever)
    الجاهز للاستخدام.
    """
    try:
        # 1. تهيئة قاعدة البيانات المتجهة (للبحث)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings_model(),
            persist_directory=VECTORSTORE_PATH,
        )

        # 2. تهيئة مخزن المستندات (لجلب المحتوى الأصلي)
        docstore = LocalFileStore(DOCSTORE_PATH)

        # 3. إنشاء المسترجع الذي يربط بينهما
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id",  # المفتاح الذي يربط بين المتجهات والمستندات الأصلية
            search_kwargs={'k': 5} # جلب أفضل 5 نتائج ذات صلة
        )
        logger.info("✅ تم تهيئة المسترجع (Retriever) بنجاح.")
        return retriever

    except Exception as e:
        logger.error(f"❌ فشل في تهيئة المسترجع: {e}")
        return None

# تهيئة المسترجع عند بدء تشغيل التطبيق ليكون جاهزًا
retriever = create_retriever()

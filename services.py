import os
import hashlib
import requests
import nltk
import base64
import fitz
import json
import time
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from groq import Groq
from config import *

try:
    nltk.download('punkt_tab', quiet=True)
except:
    nltk.download('punkt', quiet=True)

class EmbeddingService:
    def __init__(self):
        print("Loading BGE-M3 embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("BGE-M3 model loaded")

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class QdrantService:
    def __init__(self):
        self.headers = {
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json"
        }

    def get_collection_name(self, notebook_id):
        return f"notebook_{notebook_id}"

    def init_collection(self, notebook_id):
        collection_name = self.get_collection_name(notebook_id)
        
        response = requests.get(f"{QDRANT_URL}/collections/{collection_name}", headers=self.headers)
        
        if response.status_code != 200:
            config = {
                "vectors": {"size": 1024, "distance": "Cosine"},
                "optimizers_config": {"default_segment_number": 2}
            }
            response = requests.put(
                f"{QDRANT_URL}/collections/{collection_name}",
                headers=self.headers,
                json=config
            )
            return response.status_code == 200
        return True

    def store_chunks(self, notebook_id, chunks, embeddings):
        collection_name = self.get_collection_name(notebook_id)
        
        response = requests.get(f"{QDRANT_URL}/collections/{collection_name}", headers=self.headers)
        start_id = 0
        if response.status_code == 200:
            start_id = response.json()["result"].get("points_count", 0)

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = {
                "id": start_id + idx,
                "vector": embedding,
                "payload": chunk
            }
            points.append(point)

        payload = {"points": points}
        response = requests.put(
            f"{QDRANT_URL}/collections/{collection_name}/points",
            headers=self.headers,
            json=payload
        )
        
        return response.status_code == 200

    def search_knowledge(self, notebook_id, query_embedding, top_k=TOP_K_RESULTS):
        collection_name = self.get_collection_name(notebook_id)
        
        payload = {
            "vector": query_embedding,
            "limit": top_k,
            "with_payload": True
        }
        
        response = requests.post(
            f"{QDRANT_URL}/collections/{collection_name}/points/search",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            results = response.json()["result"]
            return [r["payload"] for r in results]
        return []

class GroqVisionService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_VISION_MODEL
        
    def extract_text_from_image(self, image_base64: str, context: str = "") -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Extract all text from this image. Preserve structure, formatting, and layout. Return clean, readable text. {context}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=GROQ_MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq OCR error: {e}")
            return f"OCR extraction failed: {str(e)}"
    
    def batch_extract_text(self, images_base64: List[str], context: str = "") -> List[str]:
        results = []
        
        for i, image_base64 in enumerate(images_base64):
            try:
                result = self.extract_text_from_image(image_base64, context)
                results.append(result)
                
                if i < len(images_base64) - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                results.append(f"Failed to process image {i}: {str(e)}")
        
        return results

class DocumentProcessor:
    def __init__(self):
        self.groq_vision = GroqVisionService()

    def smart_chunk_text(self, text: str, source_name: str, source_type: str, page_num: int = 1, processing_method: str = "text") -> List[Dict]:
        chunks = []
        
        text = " ".join(text.split())
        
        if len(text.strip()) < MIN_CHUNK_SIZE:
            return []
        
        sentences = nltk.sent_tokenize(text)
        
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > CHUNK_SIZE:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source_name,
                        "source_type": source_type,
                        "page": page_num,
                        "processing_method": processing_method,
                        "chunk_id": f"{source_name}_p{page_num}_c{len(chunks)}_{processing_method}",
                        "word_count": len(current_chunk.split()),
                        "char_count": len(current_chunk)
                    })
                
                overlap_sentences = current_sentences[-OVERLAP_SENTENCES:] if len(current_sentences) > OVERLAP_SENTENCES else current_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence if overlap_sentences else sentence
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_sentences.append(sentence)
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "source": source_name,
                "source_type": source_type,
                "page": page_num,
                "processing_method": processing_method,
                "chunk_id": f"{source_name}_p{page_num}_c{len(chunks)}_{processing_method}",
                "word_count": len(current_chunk.split()),
                "char_count": len(current_chunk)
            })
        
        return chunks

    def assess_text_quality(self, text: str) -> Dict:
        words = text.split()
        
        quality_metrics = {
            "char_count": len(text),
            "word_count": len(words),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "has_proper_spacing": " " in text,
            "has_sentences": "." in text or "!" in text or "?" in text,
            "special_char_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
        }
        
        score = 0
        if quality_metrics["char_count"] > PDF_OCR_THRESHOLD:
            score += 40
        if quality_metrics["word_count"] > 20:
            score += 20
        if quality_metrics["avg_word_length"] > 2:
            score += 15
        if quality_metrics["has_proper_spacing"]:
            score += 10
        if quality_metrics["has_sentences"]:
            score += 10
        if quality_metrics["special_char_ratio"] < 0.3:
            score += 5
        
        quality_metrics["quality_score"] = score
        quality_metrics["needs_ocr"] = score < 70
        
        return quality_metrics

    def extract_text_primary(self, pdf_path: str, source_name: str) -> Tuple[List[Dict], List[int]]:
        chunks = []
        pages_needing_ocr = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    
                    quality = self.assess_text_quality(text)
                    
                    if not quality["needs_ocr"]:
                        page_chunks = self.smart_chunk_text(
                            text, source_name, "pdf", page_num + 1, "text_extraction"
                        )
                        chunks.extend(page_chunks)
                        print(f"Page {page_num + 1}: Using text extraction (quality score: {quality['quality_score']})")
                    else:
                        pages_needing_ocr.append(page_num + 1)
                        print(f"Page {page_num + 1}: Marked for OCR (quality score: {quality['quality_score']})")
        
        except Exception as e:
            print(f"Error in primary text extraction: {e}")
            return [], list(range(1, 100))
        
        return chunks, pages_needing_ocr

    def extract_with_groq_ocr(self, pdf_path: str, source_name: str, pages_needing_ocr: List[int]) -> List[Dict]:
        ocr_chunks = []
        
        if not pages_needing_ocr:
            return ocr_chunks
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in pages_needing_ocr:
                    page = doc.load_page(page_num - 1)
                    
                    image_list = page.get_images(full=True)
                    
                    if image_list:
                        page_text_parts = []
                        
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                                
                                context = f"This is from page {page_num} of document '{source_name}'"
                                ocr_text = self.groq_vision.extract_text_from_image(image_base64, context)
                                
                                if ocr_text and not ocr_text.startswith("OCR extraction failed"):
                                    page_text_parts.append(ocr_text)
                                    print(f"Page {page_num}, Image {img_index + 1}: OCR successful")
                                else:
                                    print(f"Page {page_num}, Image {img_index + 1}: OCR failed")
                                    
                            except Exception as e:
                                print(f"Error processing image on page {page_num}: {e}")
                                continue
                        
                        if page_text_parts:
                            combined_text = "\n\n".join(page_text_parts)
                            page_chunks = self.smart_chunk_text(
                                combined_text, source_name, "pdf", page_num, "groq_ocr"
                            )
                            ocr_chunks.extend(page_chunks)
                    else:
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            img_data = pix.tobytes("png")
                            image_base64 = base64.b64encode(img_data).decode("utf-8")
                            
                            context = f"This is page {page_num} of document '{source_name}' rendered as an image"
                            ocr_text = self.groq_vision.extract_text_from_image(image_base64, context)
                            
                            if ocr_text and not ocr_text.startswith("OCR extraction failed"):
                                page_chunks = self.smart_chunk_text(
                                    ocr_text, source_name, "pdf", page_num, "groq_page_ocr"
                                )
                                ocr_chunks.extend(page_chunks)
                                print(f"Page {page_num}: Full page OCR successful")
                            else:
                                print(f"Page {page_num}: Full page OCR failed")
                                
                        except Exception as e:
                            print(f"Error in full page OCR for page {page_num}: {e}")
                            continue
        
        except Exception as e:
            print(f"Error in Groq OCR processing: {e}")
        
        return ocr_chunks

    def process_pdf_optimized(self, file_path: str, source_name: str) -> List[Dict]:
        print(f"Starting optimized PDF processing for: {source_name}")
        start_time = time.time()
        
        try:
            print("Stage 1: Primary text extraction...")
            text_chunks, pages_needing_ocr = self.extract_text_primary(file_path, source_name)
            
            if pages_needing_ocr:
                print(f"Stage 2: OCR processing for {len(pages_needing_ocr)} pages...")
                ocr_chunks = self.extract_with_groq_ocr(file_path, source_name, pages_needing_ocr)
                text_chunks.extend(ocr_chunks)
            else:
                print("Stage 2: No OCR needed - all pages had good text quality")
            
            final_chunks = self.optimize_chunks_for_search(text_chunks)
            
            processing_time = time.time() - start_time
            print(f"PDF processing completed in {processing_time:.2f} seconds")
            print(f"Generated {len(final_chunks)} chunks ({len(text_chunks)} before optimization)")
            
            if final_chunks:
                final_chunks[0]["processing_metadata"] = {
                    "processing_time": processing_time,
                    "total_chunks": len(final_chunks),
                    "pages_with_ocr": len(pages_needing_ocr),
                    "pages_with_text": len(set(chunk["page"] for chunk in text_chunks if chunk.get("processing_method", "").startswith("text"))),
                    "processing_methods": list(set(chunk.get("processing_method", "unknown") for chunk in final_chunks))
                }
            
            return final_chunks
            
        except Exception as e:
            print(f"Error in optimized PDF processing: {e}")
            return self.fallback_text_extraction(file_path, source_name)

    def fallback_text_extraction(self, file_path: str, source_name: str) -> List[Dict]:
        print(f"Using fallback text extraction for: {source_name}")
        chunks = []
        
        try:
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():
                        page_chunks = self.smart_chunk_text(
                            text, source_name, "pdf", page_num + 1, "fallback_text"
                        )
                        chunks.extend(page_chunks)
        
        except Exception as e:
            print(f"Error in fallback text extraction: {e}")
            return []
        
        return chunks

    def optimize_chunks_for_search(self, chunks: List[Dict]) -> List[Dict]:
        optimized = []
        
        for chunk in chunks:
            text = chunk["text"].strip()
            
            if len(text) < MIN_CHUNK_SIZE:
                continue
            
            if len(text) > MAX_CHUNK_SIZE:
                sub_chunks = self.split_large_chunk(chunk)
                optimized.extend(sub_chunks)
            else:
                optimized.append(chunk)
        
        optimized = self.remove_duplicate_chunks(optimized)
        
        return optimized

    def split_large_chunk(self, chunk: Dict) -> List[Dict]:
        text = chunk["text"]
        chunks = []
        
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > MAX_CHUNK_SIZE:
                if current_chunk:
                    new_chunk = chunk.copy()
                    new_chunk["text"] = current_chunk.strip()
                    new_chunk["chunk_id"] = f"{chunk['chunk_id']}_split_{len(chunks)}"
                    chunks.append(new_chunk)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            new_chunk = chunk.copy()
            new_chunk["text"] = current_chunk.strip()
            new_chunk["chunk_id"] = f"{chunk['chunk_id']}_split_{len(chunks)}"
            chunks.append(new_chunk)
        
        return chunks

    def remove_duplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text_signature = " ".join(chunk["text"].lower().split()[:20])
            
            if text_signature not in seen_texts:
                seen_texts.add(text_signature)
                unique_chunks.append(chunk)
        
        return unique_chunks

    def process_pdf(self, file_path: str, source_name: str) -> List[Dict]:
        return self.process_pdf_optimized(file_path, source_name)

    def process_website(self, url: str) -> List[Dict]:
        chunks = []
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
            
            text = main_content.get_text()
            text = " ".join(text.split())
            
            if text.strip():
                chunks = self.smart_chunk_text(text, url, "website", 1, "web_scraping")
            
            return chunks
            
        except Exception as e:
            print(f"Error processing website: {e}")
            return []

class GroqService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def remove_duplicates(self, chunks):
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk['text'][:200]
            if chunk_text not in seen_texts:
                seen_texts.add(chunk_text)
                unique_chunks.append(chunk)
        
        return unique_chunks

    def generate_answer(self, query, context_chunks, conversation_history=None):
        context_chunks = self.remove_duplicates(context_chunks)
        
        context = ""
        citations = []
        
        for idx, chunk in enumerate(context_chunks[:8], 1):
            processing_method = chunk.get('processing_method', 'unknown')
            context += f"[{idx}] {chunk['text']}\n\n"
            citations.append(f"[{idx}] {chunk['source']} (Page {chunk.get('page', 'N/A')}) - {processing_method}")
        
        conversation_context = ""
        if conversation_history:
            recent_messages = conversation_history[-6:]
            for msg in recent_messages:
                role = "Human" if msg['type'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            conversation_context += "\n"
        
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
IMPORTANT RULES:
1. Answer ONLY using information from the context below
2. Use [1], [2], etc. to cite which context chunk you're using
3. If the context doesn't contain the answer, say "I don't have information about that in the provided documents."
4. Consider the conversation history for context but answer based on the provided context.

CONVERSATION HISTORY:
{conversation_context}

CONTEXT:
{context}

QUESTION: {query}

Provide a clear, accurate answer with citation numbers [1], [2], etc. showing which context chunks support each statement."""
        
        try:
            completion = self.client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers based on provided context with citations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            answer = completion.choices[0].message.content
            return answer, citations
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return "Sorry, I couldn't generate an answer due to an error.", []

embedding_service = EmbeddingService()
qdrant_service = QdrantService()
doc_processor = DocumentProcessor()
groq_service = GroqService()
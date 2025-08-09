import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from werkzeug.utils import secure_filename
from services import embedding_service, qdrant_service, doc_processor, groq_service
from config import *

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_user_id():
    return request.headers.get('X-User-ID')

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    user = db.users.find_one({'email': email})
    if not user:
        user_data = {
            'email': email,
            'created_at': datetime.utcnow()
        }
        result = db.users.insert_one(user_data)
        user_id = str(result.inserted_id)
    else:
        user_id = str(user['_id'])
    
    return jsonify({'userId': user_id})

@app.route('/notebooks', methods=['GET'])
def get_notebooks():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    notebooks = list(db.notebooks.find({'user_id': ObjectId(user_id)}))
    
    for notebook in notebooks:
        notebook['_id'] = str(notebook['_id'])
        notebook['user_id'] = str(notebook['user_id'])
        
        sources_count = db.sources.count_documents({'notebook_id': ObjectId(notebook['_id'])})
        notebook['sources_count'] = sources_count
    
    return jsonify({'notebooks': notebooks})

@app.route('/notebooks', methods=['POST'])
def create_notebook():
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    title = data.get('title')
    
    if not title:
        return jsonify({'error': 'Title required'}), 400
    
    notebook_data = {
        'user_id': ObjectId(user_id),
        'title': title,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    
    result = db.notebooks.insert_one(notebook_data)
    notebook_id = str(result.inserted_id)
    
    qdrant_service.init_collection(notebook_id)
    
    return jsonify({'notebookId': notebook_id})

@app.route('/notebooks/<notebook_id>', methods=['GET'])
def get_notebook(notebook_id):
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        notebook = db.notebooks.find_one({
            '_id': ObjectId(notebook_id),
            'user_id': ObjectId(user_id)
        })
        
        if not notebook:
            return jsonify({'error': 'Notebook not found'}), 404
        
        notebook['_id'] = str(notebook['_id'])
        notebook['user_id'] = str(notebook['user_id'])
        
        sources = list(db.sources.find({'notebook_id': ObjectId(notebook_id)}))
        for source in sources:
            source['_id'] = str(source['_id'])
            source['notebook_id'] = str(source['notebook_id'])
        
        messages = list(db.messages.find({'notebook_id': ObjectId(notebook_id)}).sort('timestamp', 1))
        for message in messages:
            message['_id'] = str(message['_id'])
            message['notebook_id'] = str(message['notebook_id'])
        
        return jsonify({
            'notebook': notebook,
            'sources': sources,
            'messages': messages
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/notebooks/<notebook_id>/sources', methods=['POST'])
def add_source(notebook_id):
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    notebook = db.notebooks.find_one({
        '_id': ObjectId(notebook_id),
        'user_id': ObjectId(user_id)
    })
    
    if not notebook:
        return jsonify({'error': 'Notebook not found'}), 404
    
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
            file.save(file_path)
            
            chunks = doc_processor.process_pdf_optimized(file_path, filename)
            source_name = filename
            source_type = 'pdf'
            
            os.remove(file_path)
    else:
        data = request.get_json()
        source_type = data.get('type')
        source = data.get('source')
        
        if not source_type or not source:
            return jsonify({'error': 'Type and source required'}), 400
        
        if source_type == 'website':
            chunks = doc_processor.process_website(source)
            source_name = source
        elif source_type == 'pdf':
            chunks = doc_processor.process_pdf_optimized(source, os.path.basename(source))
            source_name = os.path.basename(source)
        else:
            return jsonify({'error': 'Invalid source type'}), 400
    
    if not chunks:
        return jsonify({'error': 'Failed to process source'}), 400
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_service.encode(texts)
    
    success = qdrant_service.store_chunks(notebook_id, chunks, embeddings)
    if not success:
        return jsonify({'error': 'Failed to store embeddings'}), 500
    
    source_data = {
        'notebook_id': ObjectId(notebook_id),
        'name': source_name,
        'type': source_type,
        'source': source if source_type == 'website' else source_name,
        'chunks_count': len(chunks),
        'processing_method': getattr(chunks[0], 'processing_method', 'hybrid') if chunks else 'unknown',
        'created_at': datetime.utcnow()
    }
    
    result = db.sources.insert_one(source_data)
    source_data['_id'] = str(result.inserted_id)
    source_data['notebook_id'] = str(source_data['notebook_id'])
    
    db.notebooks.update_one(
        {'_id': ObjectId(notebook_id)},
        {'$set': {'updated_at': datetime.utcnow()}}
    )
    
    return jsonify({'source': source_data})

@app.route('/notebooks/<notebook_id>/chat', methods=['POST'])
def chat(notebook_id):
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    notebook = db.notebooks.find_one({
        '_id': ObjectId(notebook_id),
        'user_id': ObjectId(user_id)
    })
    
    if not notebook:
        return jsonify({'error': 'Notebook not found'}), 404
    
    data = request.get_json()
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'Message required'}), 400
    
    user_message_data = {
        'notebook_id': ObjectId(notebook_id),
        'type': 'user',
        'content': message,
        'timestamp': datetime.utcnow()
    }
    db.messages.insert_one(user_message_data)
    
    query_embedding = embedding_service.encode([message])[0]
    relevant_chunks = qdrant_service.search_knowledge(notebook_id, query_embedding)
    
    if not relevant_chunks:
        response = "I don't have any documents to search through. Please add some sources first."
        citations = []
    else:
        conversation_history = list(db.messages.find({
            'notebook_id': ObjectId(notebook_id)
        }).sort('timestamp', -1).limit(10))
        conversation_history.reverse()
        
        response, citations = groq_service.generate_answer(message, relevant_chunks, conversation_history)
    
    assistant_message_data = {
        'notebook_id': ObjectId(notebook_id),
        'type': 'assistant',
        'content': response,
        'citations': citations,
        'timestamp': datetime.utcnow()
    }
    db.messages.insert_one(assistant_message_data)
    
    db.notebooks.update_one(
        {'_id': ObjectId(notebook_id)},
        {'$set': {'updated_at': datetime.utcnow()}}
    )
    
    return jsonify({
        'answer': response,
        'citations': citations
    })

@app.route('/health', methods=['GET'])
def health_check():
    try:
        db.admin.command('ismaster')
        mongo_status = 'connected'
    except:
        mongo_status = 'disconnected'
    
    return jsonify({
        'status': 'healthy',
        'services': {
            'mongodb': mongo_status,
            'qdrant': 'connected',
            'groq': 'available'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
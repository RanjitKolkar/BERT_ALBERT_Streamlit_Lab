import re, io
import streamlit as st

st.set_page_config(page_title='Transformer AI Teaching Lab', page_icon='🤖', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container{padding-top:1rem}.module{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:16px;margin:8px 0;background:rgba(128,128,128,.05)}
.badge{display:inline-block;padding:3px 9px;border-radius:20px;background:rgba(0,120,255,.12);font-size:.8rem}.small{opacity:.75;font-size:.86rem}
</style>''', unsafe_allow_html=True)

DEMO_DOC='''Digital Forensic Evidence Handling — Teaching Case\n\nDigital evidence should be acquired using a documented and repeatable process. The examiner should identify the device, record its condition, preserve the original evidence, and calculate cryptographic hashes where appropriate. A forensic image should be created using a validated acquisition procedure. The working copy may then be examined while the original evidence is protected from unnecessary alteration.\n\nThe chain of custody records who collected, transferred, received, stored, examined, or otherwise handled the evidence. Each transfer should be documented with the date, time, persons involved, purpose, and relevant identifiers.\n\nA forensic report should describe the scope, tools and versions used, acquisition method, examination steps, findings, limitations, and supporting evidence. Screenshots, hash values, logs, and other reproducibility information can help another examiner understand how the findings were obtained.\n\nForensic conclusions should be expressed carefully. An examiner should distinguish between observations, interpretations, and limitations. When evidence is incomplete, the report should clearly state what could and could not be established from the available material.'''
DEMO_Q='What information should a forensic report contain?'

@st.cache_resource(show_spinner=False)
def qa_model(model_id):
    from transformers import pipeline
    return pipeline('question-answering', model=model_id, tokenizer=model_id, device=-1)

@st.cache_resource(show_spinner=False)
def ner_model(model_id):
    from transformers import pipeline
    return pipeline('ner', model=model_id, tokenizer=model_id, aggregation_strategy='simple', device=-1)

@st.cache_resource(show_spinner=False)
def embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

def extract_text(uploaded):
    data=uploaded.read(); name=uploaded.name.lower()
    if name.endswith('.txt'): return data.decode('utf-8','ignore')
    if name.endswith('.pdf'):
        from pypdf import PdfReader
        r=PdfReader(io.BytesIO(data)); return '\n\n'.join(f'[Page {i}]\n{p.extract_text() or ""}' for i,p in enumerate(r.pages,1))
    if name.endswith('.docx'):
        from docx import Document
        d=Document(io.BytesIO(data)); return '\n'.join(p.text for p in d.paragraphs)
    return ''

def chunks(text,size=900,overlap=120):
    text=re.sub(r'\s+',' ',text).strip(); out=[]; start=0
    while start<len(text):
        end=min(len(text),start+size); out.append(text[start:end])
        if end==len(text): break
        start=max(0,end-overlap)
    return out

def tfidf(query,docs,k=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not docs:return []
    v=TfidfVectorizer(stop_words='english'); m=v.fit_transform(docs+[query]); s=cosine_similarity(m[-1],m[:-1]).ravel()
    ids=s.argsort()[::-1][:k]; return [(int(i),float(s[i]),docs[i]) for i in ids]

def semantic(query,docs,k=3):
    m=embed_model(); q=m.encode([query],normalize_embeddings=True); d=m.encode(docs,normalize_embeddings=True,show_progress_bar=False); scores=(d@q[0]).tolist(); ids=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
    return [(i,float(scores[i]),docs[i]) for i in ids]

def ollama(prompt,model,url):
    import requests
    r=requests.post(url.rstrip('/')+'/api/generate',json={'model':model,'prompt':prompt,'stream':False},timeout=180); r.raise_for_status(); return r.json()['response']

def cloud(provider,prompt,key,model):
    import requests
    if provider=='OpenAI':
        r=requests.post('https://api.openai.com/v1/chat/completions',headers={'Authorization':f'Bearer {key}'},json={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':.1},timeout=120); r.raise_for_status(); return r.json()['choices'][0]['message']['content']
    if provider=='Google Gemini':
        u=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'; r=requests.post(u,json={'contents':[{'parts':[{'text':prompt}]}]},timeout=120); r.raise_for_status(); return r.json()['candidates'][0]['content']['parts'][0]['text']
    r=requests.post('https://api.anthropic.com/v1/messages',headers={'x-api-key':key,'anthropic-version':'2023-06-01','content-type':'application/json'},json={'model':model,'max_tokens':700,'messages':[{'role':'user','content':prompt}]},timeout=120); r.raise_for_status(); return r.json()['content'][0]['text']

def grounded_prompt(context,q):
    return f'''Answer using ONLY the supplied document evidence. If the answer is absent, say so. Be concise and distinguish evidence from inference.\n\nDOCUMENT EVIDENCE:\n{context}\n\nQUESTION:\n{q}'''

st.sidebar.title('🤖 AI Teaching Lab'); st.sidebar.caption('Transformers • RAG • Local LLM • Cloud LLM')
items=['🏠 Dashboard','📘 Learning','🧪 Comparison Lab','📄 Document QA','🔎 RAG Lab','💻 Local LLM','☁️ Cloud LLM','🧩 NER Lab','⚙️ Settings']
sel=st.sidebar.radio('Modules',items)
page=items.index(sel)
st.sidebar.divider(); st.sidebar.caption('Models are lazy-loaded only when required. Cached models are reused during the session.')
st.title('Transformer AI Teaching Lab')
st.caption('Panel-based laboratory for understanding and comparing Transformer and document-AI approaches.')

if page==0:
    st.subheader('Teaching dashboard')
    c=st.columns(4)
    for col,icon,title,desc in zip(c,['📘','🧪','🔎','🤖'],['Learn','Compare','Retrieve','Generate'],['Transformer concepts','Same input, different AI approaches','RAG evidence retrieval','Local and cloud generation']):
        col.markdown(f'<div class="module"><h3>{icon} {title}</h3><span class="small">{desc}</span></div>',unsafe_allow_html=True)
    st.markdown('### Common test case')
    st.info(f'**Question:** {DEMO_Q}')
    with st.expander('View fixed demonstration document'): st.write(DEMO_DOC)
    st.markdown('''### Teaching flow\n**BERT QA → Embeddings → RAG → Local LLM → Cloud LLM**\n\nThe comparison uses one document and one question so students can see exactly what changes between approaches.''')

elif page==1:
    st.subheader('📘 Learning Module')
    t=st.selectbox('Topic',['Transformers','BERT','Tokenization','Self-Attention','Embeddings','RAG','Local LLMs','Cloud LLMs','BERT QA vs RAG'])
    lessons={'Transformers':('Transformers use attention to process relationships between tokens.','Input → embeddings → attention → feed-forward layers → output.'),'BERT':('BERT is an encoder-only Transformer for language understanding.','Typical uses include NER, classification, embeddings and extractive QA.'),'Tokenization':('Tokenization converts text into model-readable tokens.','Words may be split into subwords and special tokens.'),'Self-Attention':('Each token can use information from other tokens.','Attention computes how strongly other tokens contribute.'),'Embeddings':('Embeddings turn text into vectors.','Semantically related text can have nearby vectors.'),'RAG':('Retrieval-Augmented Generation combines retrieval with generation.','Document → chunks → search → relevant evidence → LLM → answer.'),'Local LLMs':('A local LLM runs on your own machine or infrastructure.','It offers control over data and infrastructure but needs suitable hardware.'),'Cloud LLMs':('A cloud LLM is accessed through an API.','It avoids local model hosting but introduces API, privacy and governance considerations.'),'BERT QA vs RAG':('BERT QA predicts an answer span; RAG retrieves evidence and can generate an answer.','This lab demonstrates both on the same case.')}
    a,b=lessons[t]; st.markdown(f'### {t}'); st.info(a); st.write(b)

elif page==2:
    st.subheader('🧪 Comparison Lab')
    q=st.text_input('Test question',DEMO_Q); st.info('Fixed document + common question = controlled classroom comparison.')
    if st.button('▶ Run comparison',type='primary',use_container_width=True):
        bar=st.progress(0,'Preparing comparison…'); results=[]
        try:
            bar.progress(20,'Loading BERT QA…'); p=qa_model('deepset/bert-base-cased-squad2'); bar.progress(45,'Running BERT extractive QA…'); r=p(question=q,context=DEMO_DOC); results.append(('BERT QA',r['answer'],f'Span score: {r["score"]:.3f}','Extractive'))
        except Exception as e: results.append(('BERT QA','Model unavailable',str(e)[:160],'Extractive'))
        bar.progress(70,'Running fast RAG retrieval…'); rr=tfidf(q,chunks(DEMO_DOC),3); evidence='\n\n'.join(x[2] for x in rr); results.append(('Fast RAG',evidence,f'Top similarity: {rr[0][1]:.3f}' if rr else 'No match','Retrieval'))
        bar.progress(100,'Comparison complete')
        for col,item in zip(st.columns(2),results):
            with col:
                st.markdown(f'### {item[0]}'); st.markdown(f'<div class="module"><span class="badge">{item[3]}</span><br><br>{item[1]}</div>',unsafe_allow_html=True); st.caption(item[2])
        st.markdown('### What is different?'); st.write('BERT QA tries to identify an answer span. RAG first retrieves the evidence; it needs a generator such as a local or cloud LLM to turn that evidence into a conversational answer.')

elif page==3:
    st.subheader('📄 Document QA'); up=st.file_uploader('Upload PDF, DOCX or TXT',type=['pdf','docx','txt']); q=st.text_input('Question')
    if up and q and st.button('Run BERT QA',type='primary'):
        with st.spinner('Extracting document text…'): text=extract_text(up)
        st.success(f'{len(text):,} characters extracted')
        with st.spinner('Loading BERT QA model (first use can take longer)…'): p=qa_model('deepset/bert-base-cased-squad2')
        best=None
        bar=st.progress(0,'Searching document chunks…'); cs=chunks(text,3200,350)
        for i,cx in enumerate(cs):
            try:
                r=p(question=q,context=cx)
                if best is None or r['score']>best['score']: best=r
            except Exception: pass
            bar.progress((i+1)/max(1,len(cs)),f'Checked chunk {i+1}/{len(cs)}')
        if best:
            st.markdown('### Answer'); st.success(best['answer']); st.caption(f'Score: {best["score"]:.3f} (not a factual probability)')
            with st.expander('Supporting context'): st.write(best['context'])

elif page==4:
    st.subheader('🔎 RAG Laboratory'); up=st.file_uploader('Document',type=['pdf','docx','txt'],key='ragfile'); q=st.text_input('Question',key='ragq'); method=st.radio('Retrieval', ['Fast TF-IDF RAG','Semantic RAG (MiniLM)'],horizontal=True); k=st.slider('Top-K',1,5,3)
    if up and q and st.button('Retrieve evidence',type='primary'):
        with st.spinner('Extracting document…'): text=extract_text(up)
        ds=chunks(text); bar=st.progress(15,'Creating searchable chunks…')
        try:
            if method.startswith('Fast'): bar.progress(60,'Running fast retrieval…'); rr=tfidf(q,ds,k)
            else: bar.progress(30,'Loading MiniLM embeddings…'); rr=semantic(q,ds,k); bar.progress(80,'Running semantic similarity…')
            bar.progress(100,'Retrieval complete'); st.metric('Chunks indexed',len(ds))
            for n,(_,s,cx) in enumerate(rr,1):
                with st.expander(f'#{n} • similarity {s:.3f}'): st.write(cx)
        except Exception as e: st.error(str(e))

elif page==5:
    st.subheader('💻 Local LLM Laboratory'); st.caption('Optional: connect to Ollama running locally or on a lab server.')
    url=st.text_input('Ollama URL','http://localhost:11434'); model=st.text_input('Model','qwen2.5:3b'); up=st.file_uploader('Document',type=['pdf','docx','txt'],key='localfile'); q=st.text_input('Question',key='localq'); k=st.slider('Retrieved chunks',1,5,3,key='localk')
    if up and q and st.button('Run Local RAG',type='primary'):
        with st.spinner('Extracting and retrieving evidence…'): text=extract_text(up); rr=tfidf(q,chunks(text),k); ctx='\n\n'.join(x[2] for x in rr)
        try:
            with st.spinner(f'Generating with {model}…'): ans=ollama(grounded_prompt(ctx,q),model,url)
            st.markdown('### Local LLM answer'); st.success(ans)
            with st.expander('Retrieved evidence'):
                for _,s,cx in rr: st.write(f'**Similarity {s:.3f}**\n\n{cx}')
        except Exception as e: st.error(f'Could not connect to Ollama: {e}')

elif page==6:
    st.subheader('☁️ Cloud LLM Laboratory'); provider=st.selectbox('Provider',['OpenAI','Google Gemini','Anthropic']); defaults={'OpenAI':'gpt-5-mini','Google Gemini':'gemini-2.5-flash','Anthropic':'claude-3-5-haiku-latest'}; model=st.text_input('Model',defaults[provider]); keyname={'OpenAI':'OPENAI_API_KEY','Google Gemini':'GEMINI_API_KEY','Anthropic':'ANTHROPIC_API_KEY'}[provider]; key=st.text_input('API key',type='password'); up=st.file_uploader('Document',type=['pdf','docx','txt'],key='cloudfile'); q=st.text_input('Question',key='cloudq'); k=st.slider('Retrieved chunks',1,5,3,key='cloudk')
    if up and q and st.button('Run Cloud RAG',type='primary'):
        if not key: st.warning(f'Enter {keyname} or put it in Streamlit Secrets.')
        else:
            with st.spinner('Extracting and retrieving evidence…'): text=extract_text(up); rr=tfidf(q,chunks(text),k); ctx='\n\n'.join(x[2] for x in rr)
            try:
                with st.spinner(f'Generating with {provider}…'): ans=cloud(provider,grounded_prompt(ctx,q),key,model)
                st.markdown('### Cloud LLM answer'); st.success(ans)
                with st.expander('Evidence supplied to the LLM'):
                    for _,s,cx in rr: st.write(f'**Similarity {s:.3f}**\n\n{cx}')
            except Exception as e: st.error(f'Cloud request failed: {e}')

elif page==7:
    st.subheader('🧩 NER Laboratory'); model=st.selectbox('NER model',['dslim/bert-base-NER','elastic/distilbert-base-uncased-finetuned-conll03-english','Jean-Baptiste/roberta-large-ner-english']); text=st.text_area('Text','Dr. Ranjit visited NFSU Goa Campus in Ponda on 12 September 2026.',height=150)
    if st.button('Run NER',type='primary'):
        with st.spinner('Loading NER model…'): p=ner_model(model)
        with st.spinner('Detecting entities…'): es=p(text)
        st.dataframe([{'Entity':e['word'],'Label':e['entity_group'],'Score':round(e['score'],3)} for e in es],use_container_width=True) if es else st.info('No entities detected.')

else:
    st.subheader('⚙️ Settings & Performance')
    st.markdown('''### Fast-loading strategy\n- **Lazy loading:** models load only when a module uses them.\n- **Caching:** loaded models are reused with `st.cache_resource`.\n- **Fast RAG:** TF-IDF avoids downloading an embedding model.\n- **Semantic RAG:** MiniLM loads only when selected.\n- **Chunking:** large documents are processed in pieces.\n- **Progress UI:** extraction, retrieval and generation show spinners/progress.\n\n### Architecture\n`Document → chunks → retrieval → evidence → Local/Cloud LLM → grounded answer`\n\nFor Streamlit Cloud, start with Dashboard, Learning, Comparison and Fast RAG. Heavy models download only when students explicitly select them.''')

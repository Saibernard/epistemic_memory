"""
Web Chat UI — lightweight single-page chat interface served by FastAPI.

    memory-layer chat --web
    memory-layer chat --web --brain exported-brain.json --mode llm

Opens a browser-friendly chat at http://localhost:8484.
Supports full memory management: add, edit, delete, search.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List


_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory Layer Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f0f0f; color: #e0e0e0;
    height: 100vh; display: flex; flex-direction: column;
  }
  .header {
    padding: 14px 24px; background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
    display: flex; align-items: center; gap: 12px;
  }
  .header h1 { font-size: 18px; font-weight: 600; color: #fff; }
  .header .badge {
    font-size: 11px; padding: 3px 8px; border-radius: 4px;
    background: #2d5a2d; color: #7fdf7f; font-weight: 500;
  }
  .header .stats { margin-left: auto; font-size: 13px; color: #888; }
  .header .tab-btn {
    padding: 6px 14px; border-radius: 6px; border: 1px solid #333;
    background: transparent; color: #999; font-size: 13px; cursor: pointer;
  }
  .header .tab-btn.active { background: #2a5a8a; color: #fff; border-color: #2a5a8a; }

  .main { flex: 1; display: flex; overflow: hidden; }

  /* Chat panel */
  .chat-panel { flex: 1; display: flex; flex-direction: column; }
  .messages {
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .msg { max-width: 720px; line-height: 1.6; }
  .msg.user { align-self: flex-end; }
  .msg.bot { align-self: flex-start; }
  .msg .bubble {
    padding: 12px 16px; border-radius: 12px;
    font-size: 14px; white-space: pre-wrap;
  }
  .msg.user .bubble { background: #1a3a5c; color: #c0d8f0; border-bottom-right-radius: 4px; }
  .msg.bot .bubble { background: #1a1a1a; color: #d0d0d0; border-bottom-left-radius: 4px; border: 1px solid #2a2a2a; }
  .msg.system .bubble { background: #1a2a1a; color: #7fdf7f; border: 1px solid #2d5a2d; font-size: 13px; }
  .msg .sources {
    margin-top: 8px; font-size: 11px; color: #666;
    display: flex; flex-wrap: wrap; gap: 6px;
  }
  .msg .sources span {
    background: #1a1a1a; padding: 2px 8px; border-radius: 3px;
    border: 1px solid #2a2a2a;
  }
  .msg.no-answer .bubble { border-color: #5a3a1a; color: #c0a070; }
  .input-area {
    padding: 16px 24px; background: #1a1a1a;
    border-top: 1px solid #2a2a2a;
    display: flex; gap: 12px;
  }
  .input-area input {
    flex: 1; padding: 12px 16px; border-radius: 8px;
    border: 1px solid #333; background: #0f0f0f; color: #e0e0e0;
    font-size: 14px; outline: none;
  }
  .input-area input:focus { border-color: #4a7ab5; }
  .input-area button {
    padding: 12px 20px; border-radius: 8px; border: none;
    background: #2a5a8a; color: #fff; font-size: 14px;
    cursor: pointer; font-weight: 500;
  }
  .input-area button:hover { background: #3a6a9a; }
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  .typing { color: #888; font-style: italic; font-size: 13px; padding: 8px 16px; }

  /* Memory panel */
  .memory-panel {
    width: 400px; background: #141414; border-left: 1px solid #2a2a2a;
    display: none; flex-direction: column;
  }
  .memory-panel.show { display: flex; }
  .mp-header {
    padding: 14px 18px; border-bottom: 1px solid #2a2a2a;
    display: flex; align-items: center; gap: 10px;
  }
  .mp-header h2 { font-size: 15px; font-weight: 600; flex: 1; }
  .mp-header button {
    padding: 5px 12px; border-radius: 5px; border: none;
    font-size: 12px; cursor: pointer; font-weight: 500;
  }
  .btn-add { background: #2d5a2d; color: #7fdf7f; }
  .btn-add:hover { background: #3a6a3a; }
  .mp-search {
    padding: 10px 18px;
    border-bottom: 1px solid #2a2a2a;
  }
  .mp-search input {
    width: 100%; padding: 8px 12px; border-radius: 6px;
    border: 1px solid #333; background: #0f0f0f; color: #e0e0e0;
    font-size: 13px; outline: none;
  }
  .mp-list {
    flex: 1; overflow-y: auto; padding: 8px 0;
  }
  .mem-item {
    padding: 12px 18px; border-bottom: 1px solid #1e1e1e;
    cursor: default; position: relative;
  }
  .mem-item:hover { background: #1a1a1a; }
  .mem-content { font-size: 13px; line-height: 1.5; color: #ccc; margin-bottom: 6px; }
  .mem-meta {
    font-size: 11px; color: #666;
    display: flex; gap: 10px; align-items: center;
  }
  .mem-meta .tag {
    background: #1a2a3a; color: #6a9fd0; padding: 1px 6px;
    border-radius: 3px; font-size: 10px;
  }
  .mem-actions {
    position: absolute; top: 10px; right: 14px;
    display: none; gap: 4px;
  }
  .mem-item:hover .mem-actions { display: flex; }
  .mem-actions button {
    padding: 3px 8px; border-radius: 4px; border: none;
    font-size: 11px; cursor: pointer;
  }
  .btn-edit { background: #2a3a5a; color: #8ab4e0; }
  .btn-del { background: #4a2a2a; color: #e08a8a; }
  .btn-edit:hover { background: #3a4a6a; }
  .btn-del:hover { background: #5a3a3a; }

  /* Modal */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.7);
    display: none; align-items: center; justify-content: center; z-index: 100;
  }
  .modal-overlay.show { display: flex; }
  .modal {
    background: #1a1a1a; border: 1px solid #333; border-radius: 12px;
    padding: 24px; width: 480px; max-width: 90vw;
  }
  .modal h3 { font-size: 16px; margin-bottom: 16px; }
  .modal label { font-size: 12px; color: #999; display: block; margin-bottom: 4px; margin-top: 12px; }
  .modal textarea, .modal input[type="text"] {
    width: 100%; padding: 10px 12px; border-radius: 6px;
    border: 1px solid #333; background: #0f0f0f; color: #e0e0e0;
    font-size: 13px; outline: none; font-family: inherit;
  }
  .modal textarea { min-height: 100px; resize: vertical; }
  .modal textarea:focus, .modal input[type="text"]:focus { border-color: #4a7ab5; }
  .modal-btns { margin-top: 20px; display: flex; gap: 10px; justify-content: flex-end; }
  .modal-btns button {
    padding: 8px 20px; border-radius: 6px; border: none;
    font-size: 13px; cursor: pointer; font-weight: 500;
  }
  .btn-cancel { background: #333; color: #ccc; }
  .btn-save { background: #2a5a8a; color: #fff; }
  .btn-cancel:hover { background: #444; }
  .btn-save:hover { background: #3a6a9a; }
  .slider-row { display: flex; align-items: center; gap: 10px; }
  .slider-row input[type="range"] { flex: 1; }
  .slider-row .val { font-size: 12px; color: #999; min-width: 30px; }
</style>
</head>
<body>
  <div class="header">
    <h1>Memory Layer Chat</h1>
    <span class="badge" id="mode-badge">local</span>
    <button class="tab-btn active" onclick="togglePanel(false)">Chat</button>
    <button class="tab-btn" onclick="togglePanel(true)">Memories</button>
    <span class="stats" id="stats"></span>
  </div>

  <div class="main">
    <div class="chat-panel">
      <div class="messages" id="messages">
        <div class="msg bot">
          <div class="bubble">Ask me anything about the knowledge in this memory graph. Use the Memories tab to browse, add, edit, or delete memories.</div>
        </div>
      </div>
      <div class="input-area">
        <input type="text" id="input" placeholder="Ask a question..." autofocus>
        <button id="send" onclick="send()">Send</button>
      </div>
    </div>

    <div class="memory-panel" id="memPanel">
      <div class="mp-header">
        <h2>Memories</h2>
        <button class="btn-add" onclick="openAddModal()">+ Add</button>
      </div>
      <div class="mp-search">
        <input type="text" id="memSearch" placeholder="Search memories..." oninput="searchMemories()">
      </div>
      <div class="mp-list" id="memList"></div>
    </div>
  </div>

  <!-- Add/Edit Modal -->
  <div class="modal-overlay" id="modal">
    <div class="modal">
      <h3 id="modalTitle">Add Memory</h3>
      <input type="hidden" id="editId">
      <label>Content</label>
      <textarea id="modalContent" placeholder="What should be remembered?"></textarea>
      <label>Tags (comma-separated)</label>
      <input type="text" id="modalTags" placeholder="e.g. preference, project, tech">
      <label>Importance</label>
      <div class="slider-row">
        <input type="range" id="modalImportance" min="0" max="1" step="0.1" value="0.5">
        <span class="val" id="impVal">0.5</span>
      </div>
      <div class="modal-btns">
        <button class="btn-cancel" onclick="closeModal()">Cancel</button>
        <button class="btn-save" onclick="saveMemory()">Save</button>
      </div>
    </div>
  </div>

<script>
const msgsEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const memPanel = document.getElementById('memPanel');
const memList = document.getElementById('memList');
const memSearch = document.getElementById('memSearch');
const modal = document.getElementById('modal');
const tabs = document.querySelectorAll('.tab-btn');
let allMemories = [];

document.getElementById('modalImportance').addEventListener('input', e => {
  document.getElementById('impVal').textContent = e.target.value;
});

function refreshInfo() {
  fetch('/chat/info').then(r=>r.json()).then(d=>{
    document.getElementById('mode-badge').textContent = d.mode;
    document.getElementById('stats').textContent = d.total_memories + ' memories';
  });
}
refreshInfo();

function togglePanel(show) {
  tabs[0].classList.toggle('active', !show);
  tabs[1].classList.toggle('active', show);
  memPanel.classList.toggle('show', show);
  if (show) loadMemories();
}

inputEl.addEventListener('keydown', e => { if(e.key==='Enter' && !e.shiftKey) send(); });

async function send() {
  const q = inputEl.value.trim();
  if(!q) return;
  inputEl.value = '';
  sendBtn.disabled = true;

  addMsg('user', q);
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.textContent = 'Searching memories...';
  msgsEl.appendChild(typing);
  msgsEl.scrollTop = msgsEl.scrollHeight;

  try {
    const res = await fetch('/chat/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await res.json();
    typing.remove();
    addMsg('bot', data.answer, data.sources, data.has_answer);
  } catch(e) {
    typing.remove();
    addMsg('bot', 'Error: ' + e.message, [], false);
  }
  sendBtn.disabled = false;
  inputEl.focus();
}

function addMsg(role, text, sources, hasAnswer) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if(role==='bot' && hasAnswer===false) div.classList.add('no-answer');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);
  if(sources && sources.length > 0) {
    const srcDiv = document.createElement('div');
    srcDiv.className = 'sources';
    sources.forEach(s => {
      const sp = document.createElement('span');
      sp.textContent = s.id + ' (' + (s.relevance*100).toFixed(0) + '%)';
      sp.title = s.content;
      srcDiv.appendChild(sp);
    });
    div.appendChild(srcDiv);
  }
  msgsEl.appendChild(div);
  msgsEl.scrollTop = msgsEl.scrollHeight;
}

/* Memory management */

async function loadMemories() {
  const res = await fetch('/memories/list');
  allMemories = await res.json();
  renderMemories(allMemories);
}

function searchMemories() {
  const q = memSearch.value.toLowerCase().trim();
  if (!q) { renderMemories(allMemories); return; }
  const filtered = allMemories.filter(m =>
    m.content.toLowerCase().includes(q) ||
    (m.tags || []).some(t => t.toLowerCase().includes(q))
  );
  renderMemories(filtered);
}

function renderMemories(mems) {
  memList.innerHTML = '';
  if (mems.length === 0) {
    memList.innerHTML = '<div style="padding:24px;text-align:center;color:#666;font-size:13px;">No memories found.</div>';
    return;
  }
  mems.forEach(m => {
    const el = document.createElement('div');
    el.className = 'mem-item';
    const tags = (m.tags||[]).map(t => '<span class="tag">'+esc(t)+'</span>').join(' ');
    const strength = (m.strength * 100).toFixed(0);
    el.innerHTML =
      '<div class="mem-content">' + esc(m.content) + '</div>' +
      '<div class="mem-meta">' +
        '<span>' + m.memory_type + '</span>' +
        '<span>str: ' + strength + '%</span>' +
        '<span>imp: ' + (m.importance * 100).toFixed(0) + '%</span>' +
        tags +
      '</div>' +
      '<div class="mem-actions">' +
        '<button class="btn-edit" onclick="openEditModal(\'' + m.id + '\')">Edit</button>' +
        '<button class="btn-del" onclick="deleteMemory(\'' + m.id + '\')">Delete</button>' +
      '</div>';
    memList.appendChild(el);
  });
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function openAddModal() {
  document.getElementById('modalTitle').textContent = 'Add Memory';
  document.getElementById('editId').value = '';
  document.getElementById('modalContent').value = '';
  document.getElementById('modalTags').value = '';
  document.getElementById('modalImportance').value = 0.5;
  document.getElementById('impVal').textContent = '0.5';
  modal.classList.add('show');
  document.getElementById('modalContent').focus();
}

function openEditModal(id) {
  const m = allMemories.find(x => x.id === id);
  if (!m) return;
  document.getElementById('modalTitle').textContent = 'Edit Memory';
  document.getElementById('editId').value = id;
  document.getElementById('modalContent').value = m.content;
  document.getElementById('modalTags').value = (m.tags||[]).join(', ');
  document.getElementById('modalImportance').value = m.importance;
  document.getElementById('impVal').textContent = m.importance.toFixed(1);
  modal.classList.add('show');
  document.getElementById('modalContent').focus();
}

function closeModal() { modal.classList.remove('show'); }

async function saveMemory() {
  const id = document.getElementById('editId').value;
  const content = document.getElementById('modalContent').value.trim();
  if (!content) return;
  const tags = document.getElementById('modalTags').value
    .split(',').map(t => t.trim()).filter(Boolean);
  const importance = parseFloat(document.getElementById('modalImportance').value);

  if (id) {
    await fetch('/memories/update', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({memory_id: id, new_content: content, tags, importance})
    });
    addMsg('system', 'Memory updated.');
  } else {
    await fetch('/memories/add', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({content, tags, importance})
    });
    addMsg('system', 'Memory added: "' + content.substring(0, 80) + (content.length > 80 ? '...' : '') + '"');
  }
  closeModal();
  refreshInfo();
  loadMemories();
}

async function deleteMemory(id) {
  const m = allMemories.find(x => x.id === id);
  if (!confirm('Delete this memory?\\n\\n"' + (m ? m.content.substring(0, 100) : id) + '"')) return;
  await fetch('/memories/delete', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({memory_id: id})
  });
  addMsg('system', 'Memory deleted.');
  refreshInfo();
  loadMemories();
}

modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
</script>
</body>
</html>"""


class ChatRequest(BaseModel):
    question: str


class AddMemoryRequest(BaseModel):
    content: str
    tags: List[str] = []
    importance: float = 0.5
    namespace: Optional[str] = None


class UpdateMemoryRequest(BaseModel):
    memory_id: str
    new_content: str
    tags: Optional[List[str]] = None
    importance: Optional[float] = None


class DeleteMemoryRequest(BaseModel):
    memory_id: str
    hard: bool = False


def create_chat_app(brain, engine) -> FastAPI:
    """Create a FastAPI app for the web chat interface with memory management."""
    app = FastAPI(title="Memory Layer Chat")

    def _run_sync(func, *args, **kwargs):
        import asyncio, functools
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _CHAT_HTML

    @app.get("/chat/info")
    async def chat_info():
        stats = await _run_sync(brain.get_stats)
        return {
            "mode": engine.mode,
            "total_memories": stats.total_memories,
            "namespace": engine.namespace,
        }

    @app.post("/chat/ask")
    async def chat_ask(req: ChatRequest):
        response = await _run_sync(engine.ask, req.question)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "has_answer": response.has_answer,
            "mode": response.mode,
        }

    # -- Memory management endpoints --

    @app.get("/memories/list")
    async def list_memories():
        memories = await _run_sync(
            brain.storage.get_all_memories,
            active_only=True,
            namespace=engine.namespace,
        )
        return [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type.value,
                "strength": round(m.strength, 3),
                "importance": round(m.importance, 3),
                "tags": m.tags,
                "created_at": m.created_at,
                "access_count": m.access_count,
            }
            for m in sorted(memories, key=lambda x: x.created_at, reverse=True)
        ]

    @app.post("/memories/add")
    async def add_memory(req: AddMemoryRequest):
        ns = req.namespace or engine.namespace
        memory = await _run_sync(
            brain.remember,
            req.content,
            importance=req.importance,
            tags=req.tags,
            namespace=ns,
        )
        return {"id": memory.id, "content": memory.content, "status": "added"}

    @app.post("/memories/update")
    async def update_memory(req: UpdateMemoryRequest):
        existing = await _run_sync(brain.storage.get_memory, req.memory_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Memory not found")

        updated = await _run_sync(
            brain.correct_memory,
            req.memory_id,
            req.new_content,
            importance=req.importance,
        )
        if updated and req.tags is not None:
            updated.tags = req.tags
            await _run_sync(brain.storage.store_memory, updated)

        return {"id": updated.id if updated else req.memory_id, "status": "updated"}

    @app.post("/memories/delete")
    async def delete_memory(req: DeleteMemoryRequest):
        result = await _run_sync(brain.forget_memory, req.memory_id, hard=req.hard)
        return {"id": req.memory_id, "status": "deleted", "result": result}

    @app.post("/memories/export")
    async def export_memories():
        memories = await _run_sync(
            brain.storage.get_all_memories,
            active_only=True,
            namespace=engine.namespace,
        )
        links = await _run_sync(brain.storage.get_all_links)
        return {
            "memory_count": len(memories),
            "link_count": len(links),
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "tags": m.tags,
                    "importance": m.importance,
                    "strength": m.strength,
                    "namespace": m.namespace,
                }
                for m in memories
            ],
        }

    return app

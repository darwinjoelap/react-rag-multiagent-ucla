// ==================== CONFIGURACIÓN ====================
const API_BASE_URL = 'http://localhost:8000/api';

// ==================== ESTADO GLOBAL ====================
let allEvents = [];
let currentIteration = 0;
let totalDocuments = 0;
let startTime = null;
let timeInterval = null;

// ==================== ELEMENTOS DOM ====================
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const status = document.getElementById('status');
const statusText = document.getElementById('statusText');
const timeline = document.getElementById('timeline');
const finalAnswer = document.getElementById('finalAnswer');
const sources = document.getElementById('sources');
const sourcesList = document.getElementById('sourcesList');
const mermaidGraph = document.getElementById('mermaidGraph');

// Métricas
const metricIterations = document.getElementById('metricIterations');
const metricDocuments = document.getElementById('metricDocuments');
const metricTime = document.getElementById('metricTime');
const metricCurrentNode = document.getElementById('metricCurrentNode');

// Botones de exportación
const exportGraphBtn = document.getElementById('exportGraphBtn');
const exportTraceBtn = document.getElementById('exportTraceBtn');

// ==================== EVENT LISTENERS ====================
queryForm.addEventListener('submit', handleSubmit);
exportGraphBtn.addEventListener('click', exportGraph);
exportTraceBtn.addEventListener('click', exportTrace);

// ==================== FUNCIONES PRINCIPALES ====================

async function handleSubmit(e) {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Reset estado
    resetState();
    
    // UI feedback
    submitBtn.disabled = true;
    submitBtn.textContent = 'Procesando...';
    status.classList.remove('hidden');
    statusText.textContent = 'Conectando con el servidor...';
    
    // Iniciar timer
    startTime = Date.now();
    timeInterval = setInterval(updateTimer, 100);
    
    try {
        await connectToStream(query);
    } catch (error) {
        console.error('Error en streaming:', error);
        addTimelineEvent({
            event_type: 'error',
            error_message: error.message,
            timestamp: new Date().toISOString()
        });
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Enviar';
        status.classList.add('hidden');
        clearInterval(timeInterval);
    }
}

function resetState() {
    allEvents = [];
    currentIteration = 0;
    totalDocuments = 0;
    timeline.innerHTML = '';
    finalAnswer.innerHTML = '<p class="text-gray-500">Procesando...</p>';
    sources.classList.add('hidden');
    sourcesList.innerHTML = '';
    metricIterations.textContent = '0';
    metricDocuments.textContent = '0';
    metricTime.textContent = '0';
    metricCurrentNode.textContent = '-';
}

function updateTimer() {
    if (!startTime) return;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    metricTime.textContent = elapsed;
}

// ==================== STREAMING ====================

async function connectToStream(message) {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
            console.log('Stream completed');
            break;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const eventData = JSON.parse(line.substring(6));
                    handleEvent(eventData);
                } catch (e) {
                    console.error('Error parsing event:', e, line);
                }
            }
        }
    }
}

// ==================== MANEJO DE EVENTOS ====================

function handleEvent(event) {
    console.log('Event received:', event);
    allEvents.push(event);
    
    // Actualizar UI según tipo de evento
    switch (event.event_type) {
        case 'node_start':
            handleNodeStart(event);
            break;
        case 'node_end':
            handleNodeEnd(event);
            break;
        case 'thought':
            handleThought(event);
            break;
        case 'documents_retrieved':
            handleDocumentsRetrieved(event);
            break;
        case 'grading_result':
            handleGradingResult(event);
            break;
        case 'rewrite':
            handleRewrite(event);
            break;
        case 'final_answer':
            handleFinalAnswer(event);
            break;
        case 'done':
            handleDone(event);
            break;
        case 'error':
            handleError(event);
            break;
    }
    
    // Agregar a timeline
    addTimelineEvent(event);
}

function handleNodeStart(event) {
    statusText.textContent = `Procesando en: ${event.node_name}`;
    metricCurrentNode.textContent = event.node_name;
    currentIteration = event.iteration;
    metricIterations.textContent = currentIteration;
    
    // Actualizar grafo
    updateGraphHighlight(event.node_name);
}

function handleNodeEnd(event) {
    // Limpiar highlight del grafo
    updateGraphHighlight(null);
}

function handleThought(event) {
    statusText.textContent = `Pensamiento: ${event.action}`;
}

function handleDocumentsRetrieved(event) {
    totalDocuments = event.document_count;
    metricDocuments.textContent = totalDocuments;
    statusText.textContent = `${event.document_count} documentos recuperados`;
}

function handleGradingResult(event) {
    statusText.textContent = `Evaluación: ${event.relevant_count}/${event.total_count} relevantes`;
}

function handleRewrite(event) {
    statusText.textContent = 'Reescribiendo consulta...';
}

function handleFinalAnswer(event) {
    finalAnswer.innerHTML = `
        <div class="prose max-w-none">
            <p class="text-gray-900 leading-relaxed">${escapeHtml(event.answer)}</p>
        </div>
    `;
    
    // Mostrar fuentes
    if (event.sources && event.sources.length > 0) {
        sources.classList.remove('hidden');
        sourcesList.innerHTML = event.sources.map((source, idx) => `
            <div class="p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div class="flex justify-between items-start mb-2">
                    <span class="font-semibold text-sm text-gray-700">${idx + 1}. ${escapeHtml(source.source)}</span>
                    <span class="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                        ${(source.similarity * 100).toFixed(1)}%
                    </span>
                </div>
                <p class="text-xs text-gray-600">${escapeHtml(source.document)}</p>
            </div>
        `).join('');
    }
}

function handleDone(event) {
    const finalTime = event.total_time_seconds;
    metricTime.textContent = finalTime.toFixed(1);
    metricCurrentNode.textContent = event.success ? '✓ Completado' : '✗ Error';
    statusText.textContent = event.success ? '¡Completado!' : 'Error en procesamiento';
}

function handleError(event) {
    finalAnswer.innerHTML = `
        <div class="bg-red-50 border border-red-200 rounded-lg p-4">
            <p class="text-red-800 font-medium">Error: ${escapeHtml(event.error_message)}</p>
        </div>
    `;
}

// ==================== TIMELINE ====================

function addTimelineEvent(event) {
    const eventDiv = document.createElement('div');
    eventDiv.className = 'timeline-item';
    
    const time = new Date(event.timestamp).toLocaleTimeString();
    const badge = `<span class="event-badge event-${event.event_type}">${event.event_type}</span>`;
    
    let content = '';
    
    switch (event.event_type) {
        case 'thought':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-gray-700">Pensamiento:</p>
                    <p class="text-gray-600 mt-1">${escapeHtml(event.thought.substring(0, 100))}...</p>
                    <p class="text-blue-600 mt-1 font-semibold">Acción: ${event.action}</p>
                </div>
            `;
            break;
        case 'documents_retrieved':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-gray-700">${event.document_count} documentos recuperados</p>
                    <p class="text-gray-600 text-xs mt-1">Fuentes: ${event.sources.join(', ')}</p>
                </div>
            `;
            break;
        case 'grading_result':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-gray-700">Evaluación: ${event.relevant_count}/${event.total_count} relevantes</p>
                    <p class="text-gray-600 text-xs mt-1">Decisión: ${event.decision}</p>
                </div>
            `;
            break;
        case 'final_answer':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-green-700">Respuesta generada</p>
                    <p class="text-gray-600 text-xs mt-1">${event.sources.length} fuentes citadas</p>
                </div>
            `;
            break;
        case 'node_start':
        case 'node_end':
            content = `<p class="text-sm text-gray-700">Nodo: <span class="font-semibold">${event.node_name}</span></p>`;
            break;
        default:
            content = `<p class="text-sm text-gray-600">${event.event_type}</p>`;
    }
    
    eventDiv.innerHTML = `
        <div class="flex justify-between items-start mb-1">
            ${badge}
            <span class="text-xs text-gray-500">${time}</span>
        </div>
        ${content}
    `;
    
    timeline.appendChild(eventDiv);
    timeline.scrollTop = timeline.scrollHeight;
}

// ==================== VISUALIZACIÓN DE GRAFO ====================

function updateGraphHighlight(nodeName) {
    // Por ahora solo actualizamos en consola
    // La visualización de Mermaid es estática
    console.log('Current node:', nodeName);
}

// ==================== EXPORTACIÓN ====================

function exportGraph() {
    // Exportar como SVG/PNG del grafo Mermaid
    const svg = document.querySelector('#mermaidGraph svg');
    if (!svg) {
        alert('No hay grafo para exportar');
        return;
    }
    
    // Serializar SVG
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svg);
    const blob = new Blob([svgString], { type: 'image/svg+xml' });
    
    // Descargar
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `grafo-rag-${Date.now()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
    
    alert('Grafo exportado como SVG');
}

function exportTrace() {
    if (allEvents.length === 0) {
        alert('No hay eventos para exportar');
        return;
    }
    
    const data = {
        query: queryInput.value,
        timestamp: new Date().toISOString(),
        total_events: allEvents.length,
        total_iterations: currentIteration,
        events: allEvents
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `traza-rag-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    alert('Traza exportada como JSON');
}

// ==================== UTILIDADES ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
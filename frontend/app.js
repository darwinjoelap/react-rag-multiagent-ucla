// ==================== CONFIGURACIÓN ====================
const API_BASE_URL = 'http://localhost:8000/api';

// ==================== ESTADO GLOBAL ====================
let allEvents = [];
let currentIteration = 0;
let totalDocuments = 0;
let startTime = null;
let timeInterval = null;
let currentActiveNode = null;

// ========== ✅ CAMBIO 1: HISTORIAL CONVERSACIONAL ==========
let conversationHistory = [];  // Array de {role, content, timestamp}
// ===========================================================

// ==================== ELEMENTOS DOM ====================
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const status = document.getElementById('status');
const statusText = document.getElementById('statusText');
const timeline = document.getElementById('timeline');
const answerSection = document.getElementById('answerSection');
const finalAnswer = document.getElementById('finalAnswer');
const sources = document.getElementById('sources');
const sourcesList = document.getElementById('sourcesList');
const mermaidGraph = document.getElementById('mermaidGraph');
const activeNodeIndicator = document.getElementById('activeNodeIndicator');
const activeNodeName = document.getElementById('activeNodeName');

// Métricas
const metricIterations = document.getElementById('metricIterations');
const metricDocuments = document.getElementById('metricDocuments');
const metricTime = document.getElementById('metricTime');
const metricCurrentNode = document.getElementById('metricCurrentNode');

// Botones de exportación
const exportGraphBtn = document.getElementById('exportGraphBtn');
const exportTraceBtn = document.getElementById('exportTraceBtn');

// ========== ✅ NUEVO: Botones de conversación ==========
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const exportConversationBtn = document.getElementById('exportConversationBtn');
const metricMessages = document.getElementById('metricMessages');
// =======================================================

// ==================== EVENT LISTENERS ====================
queryForm.addEventListener('submit', handleSubmit);
exportGraphBtn.addEventListener('click', exportGraph);
exportTraceBtn.addEventListener('click', exportTrace);

// ========== ✅ NUEVO: Event listeners para botones de conversación ==========
clearHistoryBtn.addEventListener('click', clearConversation);
exportConversationBtn.addEventListener('click', exportConversation);
// ============================================================================

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
    currentActiveNode = null;
    timeline.innerHTML = '';
    answerSection.classList.remove('hidden');
    finalAnswer.innerHTML = `
        <div class="flex items-center gap-3 text-gray-400">
            <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            <p class="text-sm">Procesando tu consulta...</p>
        </div>
    `;
    sources.classList.add('hidden');
    sourcesList.innerHTML = '';
    metricIterations.textContent = '0';
    metricDocuments.textContent = '0';
    metricTime.textContent = '0.0';
    metricCurrentNode.textContent = '-';
    
    // Limpiar animaciones previas del grafo
    clearGraphAnimations();
}

function updateTimer() {
    if (!startTime) return;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    metricTime.textContent = elapsed;
}

// ==================== STREAMING ====================

async function connectToStream(message) {
    console.log('🔌 Conectando al stream...');
    
    // ========== ✅ CAMBIO 2: ENVIAR HISTORIAL ==========
    // Guardar mensaje del usuario en historial
    conversationHistory.push({
        role: "user",
        content: message,
        timestamp: new Date().toISOString()
    });
    
    // Log para debugging
    console.log('📤 Enviando query con historial:', conversationHistory.length, 'mensajes');
    // ===================================================
    
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            message,
            conversation_history: conversationHistory  // ← ENVIAR HISTORIAL
        }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    console.log('✅ Stream conectado, iniciando lectura...');
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let chunkCount = 0;

    while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
            console.log(`🏁 Stream completado. Total chunks: ${chunkCount}`);
            break;
        }

        chunkCount++;
        const chunk = decoder.decode(value);
        console.log(`📦 Chunk #${chunkCount} recibido (${chunk.length} bytes)`);
        
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const eventData = JSON.parse(line.substring(6));
                    console.log(`⚡ Evento parseado: ${eventData.event_type}`);
                    handleEvent(eventData);
                } catch (e) {
                    console.error('❌ Error parsing event:', e, line);
                }
            }
        }
    }
}

// ==================== MANEJO DE EVENTOS ====================

function handleEvent(event) {
    console.log('=== EVENT RECEIVED ===');
    console.log('Type:', event.event_type);
    console.log('Timestamp:', event.timestamp);
    console.log('Full event:', event);
    console.log('====================');
    
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
    
    // Actualizar grafo con animación
    updateGraphHighlight(event.node_name);
}

function handleNodeEnd(event) {
    // Mantener el highlight hasta que llegue el siguiente nodo
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
    // Mostrar la sección de respuesta
    answerSection.classList.remove('hidden');
    answerSection.classList.add('fade-in');
    
    finalAnswer.innerHTML = `
        <div class="prose prose-invert max-w-none fade-in">
            <p class="text-white leading-relaxed">${escapeHtml(event.answer)}</p>
        </div>
    `;
    
    // Mostrar fuentes
    if (event.sources && event.sources.length > 0) {
        sources.classList.remove('hidden');
        sources.classList.add('fade-in');
        sourcesList.innerHTML = event.sources.map((source, idx) => `
            <div class="source-item p-3 bg-gray-700/50 rounded-lg border border-gray-600">
                <div class="flex justify-between items-start mb-2">
                    <span class="font-semibold text-sm text-gray-200">${idx + 1}. ${escapeHtml(source.source)}</span>
                    <span class="text-xs px-2 py-1 bg-blue-900/50 text-blue-300 rounded">
                        ${(source.similarity * 100).toFixed(1)}%
                    </span>
                </div>
                <p class="text-xs text-gray-400">${escapeHtml(source.document)}</p>
            </div>
        `).join('');
    }
    
    // Scroll suave hacia la respuesta
    setTimeout(() => {
        answerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
    
    // ========== ✅ CAMBIO 3: GUARDAR RESPUESTA EN HISTORIAL ==========
    conversationHistory.push({
        role: "assistant",
        content: event.answer,
        timestamp: new Date().toISOString()
    });
    
    // Limitar historial a últimos 10 mensajes (5 turnos)
    if (conversationHistory.length > 10) {
        conversationHistory = conversationHistory.slice(-10);
    }
    
    console.log('💾 Respuesta guardada en historial. Total:', conversationHistory.length, 'mensajes');
    
    // ========== ✅ NUEVO: Actualizar contador de mensajes ==========
    updateMessageCounter();
    // ===============================================================
    // ==================================================================
}

function handleDone(event) {
    const finalTime = event.total_time_seconds;
    metricTime.textContent = finalTime.toFixed(1);
    metricCurrentNode.textContent = event.success ? '✓ Completado' : '✗ Error';
    statusText.textContent = event.success ? '¡Completado!' : 'Error en procesamiento';
    
    // Limpiar animaciones del grafo
    clearGraphAnimations();
}

function handleError(event) {
    answerSection.classList.remove('hidden');
    answerSection.classList.add('fade-in');
    
    finalAnswer.innerHTML = `
        <div class="bg-red-900/30 border border-red-600 rounded-lg p-4 fade-in">
            <p class="text-red-300 font-medium">Error: ${escapeHtml(event.error_message)}</p>
        </div>
    `;
    
    // Scroll hacia el error
    setTimeout(() => {
        answerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// ==================== TIMELINE ====================

function addTimelineEvent(event) {
    console.log(`📝 Agregando a timeline: ${event.event_type}`);
    
    const eventDiv = document.createElement('div');
    eventDiv.className = 'timeline-item fade-in';
    
    const time = new Date(event.timestamp).toLocaleTimeString();
    const badge = `<span class="event-badge event-${event.event_type}">${event.event_type}</span>`;
    
    let content = '';
    
    switch (event.event_type) {
        case 'thought':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-gray-300">Pensamiento:</p>
                    <p class="text-gray-400 mt-1 text-xs">${escapeHtml(event.thought.substring(0, 100))}...</p>
                    <p class="text-blue-400 mt-1 font-semibold text-xs">Acción: ${event.action}</p>
                </div>
            `;
            break;
        case 'documents_retrieved':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-gray-300">${event.document_count} documentos recuperados</p>
                    <p class="text-gray-500 text-xs mt-1">Fuentes: ${event.sources.join(', ')}</p>
                </div>
            `;
            break;
        case 'grading_result':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-gray-300">Evaluación: ${event.relevant_count}/${event.total_count} relevantes</p>
                    <p class="text-gray-500 text-xs mt-1">Decisión: ${event.decision}</p>
                </div>
            `;
            break;
        case 'final_answer':
            content = `
                <div class="text-sm">
                    <p class="font-medium text-green-400">Respuesta generada</p>
                    <p class="text-gray-500 text-xs mt-1">${event.sources.length} fuentes citadas</p>
                </div>
            `;
            break;
        case 'node_start':
            content = `<p class="text-sm text-gray-300">Nodo: <span class="font-semibold text-white">${event.node_name}</span></p>`;
            break;
        case 'node_end':
            content = `<p class="text-sm text-gray-500">Finalizado: <span class="font-medium">${event.node_name}</span></p>`;
            break;
        default:
            content = `<p class="text-sm text-gray-400">${event.event_type}</p>`;
    }
    
    eventDiv.innerHTML = `
        <div class="flex justify-between items-start mb-1">
            ${badge}
            <span class="text-xs text-gray-500">${time}</span>
        </div>
        ${content}
    `;
    
    timeline.appendChild(eventDiv);
    
    // Scroll automático al último evento
    timeline.scrollTop = timeline.scrollHeight;
    
    console.log(`✅ Evento agregado a timeline. Total eventos: ${timeline.children.length}`);
}

// ==================== VISUALIZACIÓN DE GRAFO ====================

// Mapeo de nombres de nodos a IDs en el grafo Mermaid
const nodeMapping = {
    'coordinator': 'COORD',
    'search': 'SEARCH',
    'grader': 'GRADER',
    'rewriter': 'REWRITE',
    'answer': 'ANSWER'
};

// Colores originales de cada nodo (para restaurar)
const originalNodeColors = {
    'COORD': '#713f12',
    'SEARCH': '#14532d',
    'GRADER': '#831843',
    'REWRITE': '#581c87',
    'ANSWER': '#134e4a'
};

// Color para nodo activo (amarillo brillante)
const activeNodeColor = '#fbbf24';
const activeNodeStroke = '#f59e0b';
const activeNodeStrokeWidth = '4px';

function updateGraphHighlight(nodeName) {
    console.log(`🎨 updateGraphHighlight called with: "${nodeName}"`);
    
    // Limpiar highlights anteriores
    clearGraphAnimations();
    
    if (!nodeName) {
        console.log('⚠️ No nodeName provided, skipping highlight');
        return;
    }
    
    // Obtener el ID del nodo en el grafo
    const nodeId = nodeMapping[nodeName.toLowerCase()];
    if (!nodeId) {
        console.warn(`❌ Node not found in mapping: "${nodeName}"`);
        console.log('Available mappings:', nodeMapping);
        return;
    }
    
    console.log(`✅ Mapped "${nodeName}" → "${nodeId}"`);
    currentActiveNode = nodeId;
    
    // Esperar un momento para que Mermaid termine de renderizar
    setTimeout(() => {
        console.log(`⏱️ Highlighting node "${nodeId}" after 100ms delay...`);
        highlightNodeInGraph(nodeId);
    }, 100);
}

function highlightNodeInGraph(nodeId) {
    const graphContainer = document.querySelector('#mermaidGraph');
    if (!graphContainer) return;
    
    const svg = graphContainer.querySelector('svg');
    if (!svg) return;
    
    // Mostrar indicador visual del nodo activo
    if (activeNodeIndicator && activeNodeName) {
        activeNodeIndicator.classList.remove('hidden');
        activeNodeName.textContent = nodeId;
    }
    
    // Buscar todos los nodos
    const nodes = svg.querySelectorAll('.node');
    
    nodes.forEach(node => {
        const nodeText = node.querySelector('text');
        if (!nodeText) return;
        
        const text = nodeText.textContent.trim();
        
        // Verificar si es el nodo que queremos resaltar
        if (text === nodeId || 
            text.toLowerCase().includes(nodeId.toLowerCase()) ||
            nodeId.toLowerCase().includes(text.toLowerCase())) {
            
            // Buscar la forma del nodo (rect, circle, ellipse, polygon)
            const nodeShape = node.querySelector('rect, circle, ellipse, polygon');
            
            if (nodeShape) {
                // Guardar el color original si no está guardado
                if (!nodeShape.dataset.originalFill) {
                    nodeShape.dataset.originalFill = nodeShape.getAttribute('fill') || 
                                                     originalNodeColors[nodeId] || 
                                                     '#ffffff';
                    nodeShape.dataset.originalStroke = nodeShape.getAttribute('stroke') || '#000000';
                    nodeShape.dataset.originalStrokeWidth = nodeShape.getAttribute('stroke-width') || '1px';
                }
                
                // Aplicar estilo de nodo activo
                nodeShape.setAttribute('fill', activeNodeColor);
                nodeShape.setAttribute('stroke', activeNodeStroke);
                nodeShape.setAttribute('stroke-width', activeNodeStrokeWidth);
                nodeShape.classList.add('node-active');
                
                // También cambiar el color del texto a oscuro para mejor contraste
                if (nodeText) {
                    if (!nodeText.dataset.originalFill) {
                        nodeText.dataset.originalFill = nodeText.getAttribute('fill') || '#ffffff';
                    }
                    nodeText.setAttribute('fill', '#78350f');
                    nodeText.style.fontWeight = 'bold';
                }
                
                console.log(`✓ Node ${nodeId} highlighted with color ${activeNodeColor}`);
            }
        }
    });
}

function clearGraphAnimations() {
    const graphContainer = document.querySelector('#mermaidGraph');
    if (!graphContainer) return;
    
    const svg = graphContainer.querySelector('svg');
    if (!svg) return;
    
    // Ocultar indicador visual
    if (activeNodeIndicator) {
        activeNodeIndicator.classList.add('hidden');
    }
    
    // Restaurar todos los nodos a su color original
    const nodes = svg.querySelectorAll('.node');
    
    nodes.forEach(node => {
        const nodeShape = node.querySelector('rect, circle, ellipse, polygon');
        const nodeText = node.querySelector('text');
        
        if (nodeShape && nodeShape.dataset.originalFill) {
            // Restaurar colores originales
            nodeShape.setAttribute('fill', nodeShape.dataset.originalFill);
            nodeShape.setAttribute('stroke', nodeShape.dataset.originalStroke);
            nodeShape.setAttribute('stroke-width', nodeShape.dataset.originalStrokeWidth);
            nodeShape.classList.remove('node-active');
        }
        
        if (nodeText && nodeText.dataset.originalFill) {
            nodeText.setAttribute('fill', nodeText.dataset.originalFill);
            nodeText.style.fontWeight = 'normal';
        }
    });
    
    currentActiveNode = null;
    console.log('Graph animations cleared');
}

// ==================== EXPORTACIÓN ====================

function exportGraph() {
    // Exportar como SVG del grafo Mermaid
    const svg = document.querySelector('#mermaidGraph svg');
    if (!svg) {
        alert('No hay grafo para exportar');
        return;
    }
    
    // Clonar SVG para limpiar animaciones antes de exportar
    const svgClone = svg.cloneNode(true);
    svgClone.querySelectorAll('.node-active, .edge-active').forEach(el => {
        el.classList.remove('node-active', 'edge-active');
    });
    
    // Serializar SVG
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgClone);
    const blob = new Blob([svgString], { type: 'image/svg+xml' });
    
    // Descargar
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `grafo-rag-${Date.now()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
    
    // Feedback visual
    const originalText = exportGraphBtn.textContent;
    exportGraphBtn.textContent = '✓ Exportado';
    exportGraphBtn.classList.add('bg-green-700');
    
    setTimeout(() => {
        exportGraphBtn.textContent = originalText;
        exportGraphBtn.classList.remove('bg-green-700');
    }, 2000);
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
        total_documents: totalDocuments,
        conversation_history: conversationHistory,  // ← INCLUIR HISTORIAL
        events: allEvents
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `traza-rag-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    // Feedback visual
    const originalText = exportTraceBtn.textContent;
    exportTraceBtn.textContent = '✓ Exportado';
    exportTraceBtn.classList.add('bg-green-700');
    
    setTimeout(() => {
        exportTraceBtn.textContent = originalText;
        exportTraceBtn.classList.remove('bg-green-700');
    }, 2000);
}

// ==================== UTILIDADES ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== ✅ NUEVAS FUNCIONES DE CONVERSACIÓN ====================

function updateMessageCounter() {
    /**
     * Actualizar contador de mensajes en el historial
     */
    if (metricMessages) {
        metricMessages.textContent = conversationHistory.length;
    }
}

function clearConversation() {
    /**
     * Limpiar el historial de conversación
     */
    if (conversationHistory.length === 0) {
        alert('No hay conversación para limpiar.');
        return;
    }
    
    // Confirmar con el usuario
    const confirmed = confirm(
        `¿Estás seguro de que quieres limpiar la conversación?\n\n` +
        `Se borrarán ${conversationHistory.length} mensajes del historial.\n\n` +
        `El próximo mensaje iniciará una nueva conversación.`
    );
    
    if (!confirmed) return;
    
    // Limpiar historial
    conversationHistory = [];
    updateMessageCounter();
    
    console.log('🗑️ Conversación limpiada');
    
    // Feedback visual
    const originalText = clearHistoryBtn.textContent;
    const originalClass = clearHistoryBtn.className;
    
    clearHistoryBtn.textContent = '✓ Conversación Limpiada';
    clearHistoryBtn.className = clearHistoryBtn.className.replace('bg-red-600', 'bg-green-600');
    
    setTimeout(() => {
        clearHistoryBtn.textContent = originalText;
        clearHistoryBtn.className = originalClass;
    }, 2000);
}

function exportConversation() {
    /**
     * Exportar la conversación completa como JSON
     */
    if (conversationHistory.length === 0) {
        alert('No hay conversación para exportar.');
        return;
    }
    
    // Preparar datos de exportación
    const exportData = {
        export_type: "conversation",
        export_date: new Date().toISOString(),
        total_messages: conversationHistory.length,
        conversation_history: conversationHistory,
        metadata: {
            user_agent: navigator.userAgent,
            platform: navigator.platform,
            screen_resolution: `${window.screen.width}x${window.screen.height}`
        }
    };
    
    // Crear blob y descargar
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    
    // Nombre del archivo con timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
    a.download = `conversacion-rag-${timestamp}.json`;
    
    a.click();
    URL.revokeObjectURL(url);
    
    console.log(`📥 Conversación exportada: ${conversationHistory.length} mensajes`);
    
    // Feedback visual
    const originalText = exportConversationBtn.textContent;
    const originalClass = exportConversationBtn.className;
    
    exportConversationBtn.innerHTML = `
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
        </svg>
        Exportado
    `;
    exportConversationBtn.className = exportConversationBtn.className.replace('bg-blue-600', 'bg-green-600');
    
    setTimeout(() => {
        exportConversationBtn.innerHTML = originalText;
        exportConversationBtn.className = originalClass;
    }, 2000);
}

// ==============================================================================

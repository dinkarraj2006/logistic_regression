// script.js - Logistic Regression Learning Lab
// Complete implementation from scratch

// ==================== GLOBAL STATE ====================
let rawData = null;              // original parsed CSV rows
let processedData = null;       // after preprocessing (numeric matrix)
let featureNames = [];
let targetCol = null;
let uniqueTargets = [];
let X_train = [], X_test = [], y_train = [], y_test = [];
let modelWeights = null;
let modelBias = null;
let trained = false;
let scalerParams = { mean: [], std: [] };
let targetMapping = {};

// ==================== UTILITY FUNCTIONS ====================
function showMessage(elementId, message, isError = false) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = isError ? `<div class="error-message">⚠️ ${message}</div>` : `<div class="success-message">✅ ${message}</div>`;
        setTimeout(() => {
            if (el.innerHTML.includes(message)) el.innerHTML = '';
        }, 5000);
    }
}

// ==================== CSV UPLOAD & DATASET HANDLING ====================
document.getElementById('csvUpload').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    Papa.parse(file, { 
        header: true, 
        dynamicTyping: true, 
        skipEmptyLines: true,
        complete: function(results) {
            rawData = results.data.filter(row => Object.values(row).some(v => v !== null && v !== ""));
            if (rawData.length === 0) {
                alert("Empty CSV file!");
                return;
            }
            featureNames = Object.keys(rawData[0]);
            
            // Display preview
            let previewHtml = `<div style="overflow-x: auto;"><table><thead><tr>${featureNames.map(f => `<th>${f}</th>`).join('')}</tr></thead><tbody>`;
            rawData.slice(0, 5).forEach(row => {
                previewHtml += `<tr>${featureNames.map(f => `<td>${row[f] ?? ''}</td>`).join('')}</tr>`;
            });
            previewHtml += `</tbody></table></div><p><strong>Shape:</strong> ${rawData.length} rows, ${featureNames.length} columns</p>`;
            document.getElementById('previewTable').innerHTML = previewHtml;
            
            // Create target selection dropdown
            let selectHtml = `
                <div class="card-title">Select Target Column</div>
                <select id="targetSelect" style="width: 100%; margin-bottom: 10px;">
                    ${featureNames.map(f => `<option value="${f}">${f}</option>`).join('')}
                </select>
                <button id="confirmTarget">Confirm Target</button>
                <div class="info-tooltip" style="margin-top: 10px;">
                    <i class="fas fa-info-circle"></i> Target should be binary or multi-class categorical.
                </div>
            `;
            document.getElementById('targetSelectBox').innerHTML = selectHtml;
            
            document.getElementById('confirmTarget').onclick = () => {
                targetCol = document.getElementById('targetSelect').value;
                showMessage('targetSelectBox', `Target set to "${targetCol}". You can now proceed to preprocessing.`);
                // Mark step 1 as ready
                document.querySelector('[data-step="1"]').click();
            };
        }
    });
});

// ==================== PREPROCESSING ====================
function preprocessData() {
    if (!rawData || !targetCol) {
        alert("Please upload a dataset and select target column first!");
        return null;
    }
    
    const features = featureNames.filter(f => f !== targetCol);
    let X = rawData.map(row => features.map(f => row[f]));
    let y = rawData.map(row => row[targetCol]);
    
    // Handle missing values (impute with mean for numeric columns)
    if (document.getElementById('handleMissing').checked) {
        for (let j = 0; j < features.length; j++) {
            const colVals = X.map(row => row[j]).filter(v => typeof v === 'number' && !isNaN(v));
            const meanVal = colVals.length > 0 ? colVals.reduce((a, b) => a + b, 0) / colVals.length : 0;
            for (let i = 0; i < X.length; i++) {
                if (X[i][j] === undefined || X[i][j] === null || isNaN(X[i][j])) {
                    X[i][j] = meanVal;
                }
            }
        }
    }
    
    // Encode target variable to numeric
    const uniqueY = [...new Set(y)];
    uniqueTargets = uniqueY;
    targetMapping = {};
    uniqueY.forEach((val, idx) => { targetMapping[val] = idx; });
    const yNumeric = y.map(v => targetMapping[v]);
    
    // Encode categorical features (simple label encoding)
    if (document.getElementById('labelEncode').checked) {
        for (let j = 0; j < features.length; j++) {
            const colVals = X.map(row => row[j]);
            if (colVals.some(v => typeof v === 'string')) {
                const cats = [...new Set(colVals)];
                const catMap = {};
                cats.forEach((c, idx) => { catMap[c] = idx; });
                for (let i = 0; i < X.length; i++) {
                    X[i][j] = catMap[X[i][j]];
                }
            }
        }
    }
    
    // Feature scaling (Standardization)
    let Xscaled = X.map(row => [...row]);
    if (document.getElementById('scaleFeatures').checked) {
        scalerParams.mean = [];
        scalerParams.std = [];
        for (let j = 0; j < features.length; j++) {
            const col = Xscaled.map(row => row[j]);
            const mean = col.reduce((a, b) => a + b, 0) / col.length;
            const variance = col.map(v => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / col.length;
            const std = Math.sqrt(variance) || 1;
            scalerParams.mean[j] = mean;
            scalerParams.std[j] = std;
            for (let i = 0; i < Xscaled.length; i++) {
                Xscaled[i][j] = (Xscaled[i][j] - mean) / std;
            }
        }
    }
    
    processedData = { 
        X: Xscaled, 
        y: yNumeric, 
        featureNames: features, 
        targetLabels: uniqueTargets,
        originalX: X
    };
    
    document.getElementById('compareMsg').innerHTML = `
        <strong>✅ Preprocessing Complete!</strong><br>
        Original: ${rawData.length} rows × ${features.length} features<br>
        Processed: ${Xscaled.length} samples, features scaled to zero mean & unit variance<br>
        Target classes: ${uniqueTargets.join(', ')}
    `;
    
    return processedData;
}

document.getElementById('applyPreprocessBtn').onclick = () => {
    const result = preprocessData();
    if (result) {
        showMessage('preprocessCompare', 'Preprocessing applied successfully! Ready for EDA and Training.');
    }
};

// ==================== EXPLORATORY DATA ANALYSIS ====================
function renderEDA() {
    if (!processedData) {
        alert("Please apply preprocessing first!");
        return;
    }
    
    const { X, y, featureNames: featNames, targetLabels } = processedData;
    
    // 1. Class Distribution
    const counts = targetLabels.map((_, idx) => y.filter(v => v === idx).length);
    Plotly.newPlot('classDistChart', [{
        x: targetLabels.map(l => String(l)),
        y: counts,
        type: 'bar',
        marker: { color: '#3b82f6' }
    }], {
        title: 'Class Distribution',
        xaxis: { title: 'Class Label' },
        yaxis: { title: 'Count' }
    });
    
    // 2. Feature Distributions (first 3 features)
    const traces = featNames.slice(0, Math.min(3, featNames.length)).map((f, idx) => ({
        y: X.map(row => row[idx]),
        name: f,
        type: 'box',
        boxmean: 'sd'
    }));
    Plotly.newPlot('featureDistChart', traces, {
        title: 'Feature Distributions (Box Plot)',
        yaxis: { title: 'Standardized Value' }
    });
    
    // 3. Correlation Matrix (for first 5 features)
    const nFeat = Math.min(5, featNames.length);
    const corrMatrix = [];
    for (let i = 0; i < nFeat; i++) {
        corrMatrix.push([]);
        for (let j = 0; j < nFeat; j++) {
            const col1 = X.map(row => row[i]);
            const col2 = X.map(row => row[j]);
            const mean1 = col1.reduce((a, b) => a + b, 0) / col1.length;
            const mean2 = col2.reduce((a, b) => a + b, 0) / col2.length;
            let num = 0, d1 = 0, d2 = 0;
            for (let k = 0; k < col1.length; k++) {
                num += (col1[k] - mean1) * (col2[k] - mean2);
                d1 += Math.pow(col1[k] - mean1, 2);
                d2 += Math.pow(col2[k] - mean2, 2);
            }
            corrMatrix[i][j] = num / Math.sqrt(d1 * d2);
        }
    }
    Plotly.newPlot('corrPlot', [{
        z: corrMatrix,
        x: featNames.slice(0, nFeat),
        y: featNames.slice(0, nFeat),
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0
    }], { title: 'Feature Correlation Matrix' });
    
    // 4. Feature vs Target (box plot for first feature)
    const firstFeat = featNames[0];
    const class0Vals = X.filter((_, i) => y[i] === 0).map(row => row[0]);
    const class1Vals = X.filter((_, i) => y[i] === 1).map(row => row[0]);
    Plotly.newPlot('featureTargetPlot', [
        { y: class0Vals, name: `Class ${targetLabels[0]}`, type: 'box', boxmean: 'sd' },
        { y: class1Vals, name: `Class ${targetLabels[1] || 'Other'}`, type: 'box', boxmean: 'sd' }
    ], { title: `${firstFeat} Distribution by Target Class` });
}

document.getElementById('refreshEdaBtn').onclick = renderEDA;

// ==================== TRAIN-TEST SPLIT ====================
document.getElementById('splitDataBtn').onclick = () => {
    if (!processedData) {
        alert("Please preprocess the data first!");
        return;
    }
    
    const { X, y } = processedData;
    const testRatio = parseFloat(document.getElementById('testRatio').value);
    const splitIdx = Math.floor(X.length * (1 - testRatio));
    
    X_train = X.slice(0, splitIdx);
    y_train = y.slice(0, splitIdx);
    X_test = X.slice(splitIdx);
    y_test = y.slice(splitIdx);
    
    document.getElementById('splitInfo').innerHTML = `
        <div class="success-message">
            ✅ Train set: ${X_train.length} samples<br>
            ✅ Test set: ${X_test.length} samples
        </div>
    `;
};

// ==================== LOGISTIC REGRESSION FROM SCRATCH ====================
function sigmoid(z) {
    // Clipping to avoid overflow
    const clipped = Math.min(20, Math.max(-20, z));
    return 1 / (1 + Math.exp(-clipped));
}

function trainLogisticRegression(X, y, learningRate = 0.1, epochs = 1000, verbose = true) {
    const n = X.length;
    const m = X[0].length;
    let weights = new Array(m).fill(0);
    let bias = 0;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        let dw = new Array(m).fill(0);
        let db = 0;
        
        // Gradient descent
        for (let i = 0; i < n; i++) {
            let z = bias;
            for (let j = 0; j < m; j++) {
                z += weights[j] * X[i][j];
            }
            const pred = sigmoid(z);
            const error = pred - y[i];
            
            for (let j = 0; j < m; j++) {
                dw[j] += error * X[i][j];
            }
            db += error;
        }
        
        // Update parameters
        for (let j = 0; j < m; j++) {
            weights[j] -= learningRate * dw[j] / n;
        }
        bias -= learningRate * db / n;
        
        // Log loss every 200 epochs
        if (verbose && epoch % 200 === 0) {
            let loss = 0;
            for (let i = 0; i < n; i++) {
                let z = bias;
                for (let j = 0; j < m; j++) z += weights[j] * X[i][j];
                const pred = sigmoid(z);
                loss += -y[i] * Math.log(pred + 1e-8) - (1 - y[i]) * Math.log(1 - pred + 1e-8);
            }
            console.log(`Epoch ${epoch}, Loss: ${(loss / n).toFixed(4)}`);
        }
    }
    
    return { weights, bias };
}

function predictProba(sample) {
    if (!modelWeights || modelBias === null) return 0;
    let z = modelBias;
    for (let i = 0; i < modelWeights.length; i++) {
        z += modelWeights[i] * sample[i];
    }
    return sigmoid(z);
}

function predict(sample, threshold = 0.5) {
    const prob = predictProba(sample);
    return prob >= threshold ? 1 : 0;
}

document.getElementById('trainModelBtn').onclick = async () => {
    if (!X_train.length) {
        alert("Please split the data first!");
        return;
    }
    
    document.getElementById('trainingStatus').innerHTML = '<div class="info-tooltip">⏳ Training in progress...</div>';
    
    // Run training asynchronously to not block UI
    setTimeout(() => {
        const { weights, bias } = trainLogisticRegression(X_train, y_train, 0.1, 800);
        modelWeights = weights;
        modelBias = bias;
        trained = true;
        
        document.getElementById('paramDisplay').innerHTML = `
            <div class="math-block">
                <strong>📊 Learned Parameters:</strong><br>
                Weights: [${weights.map(w => w.toFixed(4)).join(', ')}]<br>
                Bias: ${bias.toFixed(4)}
            </div>
            <div class="info-tooltip">
                <i class="fas fa-chart-line"></i> Decision boundary: w·x + b = 0
            </div>
        `;
        document.getElementById('trainingStatus').innerHTML = '<div class="success-message">✅ Training complete! Model ready for predictions.</div>';
        document.getElementById('showDemoCalc').disabled = false;
        
        // Auto-evaluate if test set exists
        if (X_test.length) {
            evaluateModel();
        }
    }, 100);
};

// ==================== K-FOLD CROSS-VALIDATION ====================
function kFoldCrossValidation(X, y, k, learningRate = 0.1, epochs = 500) {
    const foldSize = Math.floor(X.length / k);
    const accuracies = [];
    
    for (let fold = 0; fold < k; fold++) {
        const start = fold * foldSize;
        const end = (fold + 1) * foldSize;
        
        const X_val = X.slice(start, end);
        const y_val = y.slice(start, end);
        const X_train_fold = [...X.slice(0, start), ...X.slice(end)];
        const y_train_fold = [...y.slice(0, start), ...y.slice(end)];
        
        const { weights, bias } = trainLogisticRegression(X_train_fold, y_train_fold, learningRate, epochs, false);
        
        let correct = 0;
        for (let i = 0; i < X_val.length; i++) {
            let z = bias;
            for (let j = 0; j < weights.length; j++) z += weights[j] * X_val[i][j];
            const pred = sigmoid(z) >= 0.5 ? 1 : 0;
            if (pred === y_val[i]) correct++;
        }
        accuracies.push(correct / X_val.length);
    }
    
    const meanAcc = accuracies.reduce((a, b) => a + b, 0) / k;
    const stdAcc = Math.sqrt(accuracies.map(a => Math.pow(a - meanAcc, 2)).reduce((a, b) => a + b, 0) / k);
    return { mean: meanAcc, std: stdAcc, accuracies };
}

document.getElementById('runCvBtn').onclick = () => {
    if (!processedData) {
        alert("Please preprocess the data first!");
        return;
    }
    
    const k = parseInt(document.getElementById('kfoldVal').value);
    const { X, y } = processedData;
    
    document.getElementById('cvResult').innerHTML = '<div class="info-tooltip">⏳ Running cross-validation...</div>';
    
    setTimeout(() => {
        const { mean, std, accuracies } = kFoldCrossValidation(X, y, k);
        document.getElementById('cvResult').innerHTML = `
            <div class="success-message">
                <strong>📊 ${k}-Fold Cross-Validation Results:</strong><br>
                Mean Accuracy: ${(mean * 100).toFixed(2)}% ± ${(std * 100).toFixed(2)}%<br>
                Individual folds: ${accuracies.map(acc => (acc * 100).toFixed(1) + '%').join(', ')}
            </div>
        `;
    }, 100);
};

// ==================== EVALUATION METRICS ====================
function evaluateModel() {
    if (!trained || X_test.length === 0) {
        document.getElementById('metrics').innerHTML = '⚠️ Train model and split data first.';
        return;
    }
    
    const predictions = X_test.map(x => predict(x));
    const trueY = y_test;
    
    // Calculate metrics
    const n = trueY.length;
    const accuracy = predictions.filter((p, i) => p === trueY[i]).length / n;
    
    const tp = predictions.filter((p, i) => p === 1 && trueY[i] === 1).length;
    const fp = predictions.filter((p, i) => p === 1 && trueY[i] === 0).length;
    const fn = predictions.filter((p, i) => p === 0 && trueY[i] === 1).length;
    const tn = predictions.filter((p, i) => p === 0 && trueY[i] === 0).length;
    
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * precision * recall / (precision + recall) || 0;
    
    document.getElementById('metrics').innerHTML = `
        <div class="success-message">
            <strong>📈 Evaluation Metrics:</strong><br>
            Accuracy: ${(accuracy * 100).toFixed(2)}%<br>
            Precision: ${(precision * 100).toFixed(2)}%<br>
            Recall: ${(recall * 100).toFixed(2)}%<br>
            F1-Score: ${(f1 * 100).toFixed(2)}%
        </div>
    `;
    
    // Confusion Matrix
    const cm = [[tn, fp], [fn, tp]];
    Plotly.newPlot('confMatrixPlot', [{
        z: cm,
        x: ['Predicted 0', 'Predicted 1'],
        y: ['Actual 0', 'Actual 1'],
        type: 'heatmap',
        colorscale: 'Blues',
        text: cm.map(row => row.map(String)),
        texttemplate: '%{text}',
        showscale: true
    }], { title: 'Confusion Matrix' });
    
    // ROC Curve
    const thresholds = [];
    const tprList = [];
    const fprList = [];
    for (let th = 0; th <= 1; th += 0.05) {
        thresholds.push(th);
        const predsTh = X_test.map(x => predict(x, th));
        const tpTh = predsTh.filter((p, i) => p === 1 && trueY[i] === 1).length;
        const fpTh = predsTh.filter((p, i) => p === 1 && trueY[i] === 0).length;
        const fnTh = predsTh.filter((p, i) => p === 0 && trueY[i] === 1).length;
        const tnTh = predsTh.filter((p, i) => p === 0 && trueY[i] === 0).length;
        tprList.push(tpTh / (tpTh + fnTh) || 0);
        fprList.push(fpTh / (fpTh + tnTh) || 0);
    }
    
    // Calculate AUC approximation
    let auc = 0;
    for (let i = 0; i < tprList.length - 1; i++) {
        auc += (fprList[i + 1] - fprList[i]) * (tprList[i] + tprList[i + 1]) / 2;
    }
    
    Plotly.newPlot('rocPlot', [{
        x: fprList,
        y: tprList,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'ROC Curve',
        line: { color: '#3b82f6', width: 2 }
    }, {
        x: [0, 1],
        y: [0, 1],
        type: 'scatter',
        mode: 'lines',
        name: 'Random Classifier',
        line: { color: 'gray', dash: 'dash' }
    }], {
        title: `ROC Curve (AUC = ${auc.toFixed(3)})`,
        xaxis: { title: 'False Positive Rate' },
        yaxis: { title: 'True Positive Rate' }
    });
    
    // Decision Boundary for 2D features
    if (processedData.featureNames.length === 2 && X_test.length > 0) {
        const xGrid = [];
        const yGrid = [];
        const zGrid = [];
        for (let x = -2.5; x <= 2.5; x += 0.15) {
            for (let y = -2.5; y <= 2.5; y += 0.15) {
                xGrid.push(x);
                yGrid.push(y);
                const prob = predictProba([x, y]);
                zGrid.push(prob);
            }
        }
        Plotly.newPlot('decisionBoundaryPlot', [{
            x: xGrid,
            y: yGrid,
            z: zGrid,
            type: 'contour',
            colorscale: 'RdBu',
            contours: { showlabels: true },
            colorbar: { title: 'Probability' }
        }], {
            title: 'Decision Boundary (Probability Contour)',
            xaxis: { title: 'Feature 1 (scaled)' },
            yaxis: { title: 'Feature 2 (scaled)' }
        });
    }
}

// ==================== CUSTOM INFERENCE ====================
document.getElementById('predictCustomBtn').onclick = () => {
    if (!trained) {
        alert("Please train the model first!");
        return;
    }
    
    const inputStr = document.getElementById('customInputFeat').value;
    if (!inputStr.trim()) {
        alert("Please enter feature values separated by commas");
        return;
    }
    
    const values = inputStr.split(',').map(v => parseFloat(v.trim()));
    if (values.length !== modelWeights.length) {
        alert(`Expected ${modelWeights.length} features, got ${values.length}`);
        return;
    }
    
    // Apply standardization if enabled
    let scaledValues = [...values];
    if (document.getElementById('scaleFeatures').checked && scalerParams.mean.length) {
        for (let i = 0; i < values.length; i++) {
            scaledValues[i] = (values[i] - scalerParams.mean[i]) / scalerParams.std[i];
        }
    }
    
    const z = modelWeights.reduce((sum, w, i) => sum + w * scaledValues[i], modelBias);
    const probability = sigmoid(z);
    const prediction = probability >= 0.5 ? 1 : 0;
    const predictedClass = uniqueTargets[prediction] || prediction;
    
    document.getElementById('inferenceSteps').innerHTML = `
        <div class="math-block">
            <strong>📐 Step-by-Step Computation:</strong><br>
            1️⃣ Linear combination: z = ${modelBias.toFixed(4)} + ${modelWeights.map((w, i) => `(${w.toFixed(4)} × ${scaledValues[i].toFixed(4)})`).join(' + ')}<br>
            2️⃣ z = ${z.toFixed(4)}<br>
            3️⃣ Sigmoid: σ(z) = 1 / (1 + e<sup>-${z.toFixed(4)}</sup>) = ${probability.toFixed(6)}<br>
            4️⃣ Probability of class 1: ${(probability * 100).toFixed(2)}%<br>
            5️⃣ <strong>Final Prediction: ${predictedClass}</strong> (threshold = 0.5)
        </div>
    `;
};

// ==================== DEMO CALCULATION ====================
document.getElementById('showDemoCalc').onclick = () => {
    if (!trained || X_test.length === 0) {
        alert("Please train the model and ensure test data exists!");
        return;
    }
    
    const sample = X_test[0];
    const trueLabel = y_test[0];
    let z = modelBias;
    const calcSteps = [];
    
    for (let i = 0; i < modelWeights.length; i++) {
        z += modelWeights[i] * sample[i];
        calcSteps.push(`${modelWeights[i].toFixed(4)} × ${sample[i].toFixed(4)}`);
    }
    
    const probability = sigmoid(z);
    const prediction = probability >= 0.5 ? 1 : 0;
    
    document.getElementById('stepCalcDemo').innerHTML = `
        <div class="math-block">
            <strong>🧪 Sample #1 Calculation:</strong><br>
            <strong>Features:</strong> [${sample.map(v => v.toFixed(4)).join(', ')}]<br>
            <strong>True Label:</strong> ${uniqueTargets[trueLabel] || trueLabel}<br><br>
            <strong>z = b + Σ(wᵢ × xᵢ)</strong><br>
            z = ${modelBias.toFixed(4)} + ${calcSteps.join(' + ')}<br>
            z = ${z.toFixed(4)}<br><br>
            <strong>σ(z) = 1 / (1 + e<sup>-${z.toFixed(4)}</sup>)</strong><br>
            Probability = ${(probability * 100).toFixed(2)}%<br><br>
            <strong>Predicted Class:</strong> ${prediction === 1 ? uniqueTargets[1] || 'Class 1' : uniqueTargets[0] || 'Class 0'}<br>
            <strong>Correct?</strong> ${prediction === trueLabel ? '✅ Yes' : '❌ No'}
        </div>
    `;
};

// ==================== STEP NAVIGATION ====================
const panels = document.querySelectorAll('.panel');
const stepBtns = document.querySelectorAll('.step-btn');

stepBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const step = parseInt(btn.dataset.step);
        panels.forEach((panel, idx) => {
            panel.classList.toggle('active-panel', idx === step);
        });
        stepBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Refresh specific panels when navigated
        if (step === 2 && processedData) {
            renderEDA();
        }
        if (step === 5 && trained) {
            evaluateModel();
        }
    });
});

console.log("🚀 Logistic Regression Learning Lab loaded successfully!");
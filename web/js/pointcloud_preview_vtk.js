/**
 * VTK.js Point Cloud Preview for ComfyUI
 * Uses VTK.js for scientific 3D visualization
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[SAM3DObjects] Loading VTK point cloud preview extension...");

// Register the extension
app.registerExtension({
    name: "SAM3DObjects.PointCloudPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SAM3D_PreviewPointCloud") {
            console.log("[SAM3DObjects] Registering Preview Point Cloud node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log('[SAM3DObjects] Creating VTK point cloud preview widget');

                // Create container div
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.position = "relative";
                container.style.background = "#1a1a1a";
                container.style.borderRadius = "4px";
                container.style.overflow = "hidden";

                // Create iframe for VTK viewer
                // Use the folder name (lowercase) for the extension path
                const iframe = document.createElement("iframe");
                iframe.src = "/extensions/comfyui-sam3dobjects/viewer_vtk.html";
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.display = "block";

                // Add load event listener for debugging
                iframe.addEventListener('load', () => {
                    console.log('[SAM3DObjects] iframe loaded successfully');
                });

                container.appendChild(iframe);

                // Store iframe reference
                this._vtkIframe = iframe;
                console.log('[SAM3DObjects] iframe created with src:', iframe.src);

                // Add widget using ComfyUI's addDOMWidget API
                const widget = this.addDOMWidget("preview", "POINTCLOUD_PREVIEW_VTK", container, {
                    getValue() { return ""; },
                    setValue(v) { 
                        // When widget value changes, try to load the point cloud
                        if (v && typeof v === 'string' && v.trim() !== '') {
                            console.log('[SAM3DObjects] Widget value changed:', v);
                            this.loadPointCloudFromPath(v);
                        }
                    }
                });
                
                // Helper function to load point cloud from file path
                this.loadPointCloudFromPath = function(filePath) {
                    if (!filePath || filePath.trim() === '') {
                        console.warn('[SAM3DObjects] Empty file path in loadPointCloudFromPath');
                        return;
                    }
                    
                    if (!this._vtkIframe) {
                        console.warn('[SAM3DObjects] iframe not available');
                        return;
                    }
                    
                    // Normalize path separators (Windows backslash to forward slash)
                    filePath = filePath.replace(/\\/g, '/').trim();
                    
                    // Extract relative path from output/input folder
                    const outputMatch = filePath.match(/(?:^|\/)(output|input)\/(.+)$/);
                    
                    let url;
                    if (outputMatch) {
                        const [, type, relativePath] = outputMatch;
                        const pathParts = relativePath.split('/');
                        const filename = pathParts.pop();
                        const subfolder = pathParts.join('/');
                        url = `/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                    } else {
                        // Fallback: try to use filename directly
                        const filename = filePath.split('/').pop();
                        url = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                        console.warn('[SAM3DObjects] Could not parse output/input path, using filename only:', filename);
                    }
                    
                    console.log('[SAM3DObjects] File path:', filePath);
                    console.log('[SAM3DObjects] Constructed URL:', url);
                    
                    // Send message to iframe
                    const sendMessage = () => {
                        console.log('[SAM3DObjects] Sending postMessage to iframe with URL:', url);
                        if (this._vtkIframe && this._vtkIframe.contentWindow) {
                            this._vtkIframe.contentWindow.postMessage({
                                type: 'loadPointCloud',
                                url: url
                            }, '*');
                            console.log('[SAM3DObjects] postMessage sent successfully');
                        } else {
                            console.warn('[SAM3DObjects] iframe or contentWindow not available');
                        }
                    };
                    
                    // Wait for iframe to be ready
                    if (this._vtkIframe.contentWindow && this._vtkIframe.contentDocument?.readyState === 'complete') {
                        sendMessage();
                    } else {
                        // Wait for iframe to load
                        const checkReady = () => {
                            if (this._vtkIframe.contentWindow && this._vtkIframe.contentDocument?.readyState === 'complete') {
                                sendMessage();
                            } else {
                                setTimeout(checkReady, 100);
                            }
                        };
                        this._vtkIframe.addEventListener('load', () => {
                            console.log('[SAM3DObjects] iframe load event fired, sending message');
                            setTimeout(sendMessage, 100); // Small delay to ensure iframe is fully ready
                        }, { once: true });
                        checkReady();
                    }
                };

                // Set widget size
                widget.computeSize = function(width) {
                    return [width || 512, (width || 512) + 60];  // Extra height for controls
                };

                widget.element = container;

                // Set initial node size
                this.setSize([512, 572]);

                console.log("[SAM3DObjects] VTK Widget created successfully");

                return r;
            };

            // Handle execution
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                console.log('[SAM3DObjects] onExecuted called, checking for message content...');
                onExecuted?.apply(this, arguments);

                console.log('[SAM3DObjects] VTK Preview node executed with message:', message);
                console.log('[SAM3DObjects] Full message object:', JSON.stringify(message, null, 2));
                
                // Try multiple ways to get the file path:
                // 1. From message.ui.file_path (ComfyUI UI data)
                // 2. From message.file_path (direct)
                // 3. From widget value
                let filePath = null;
                
                if (message?.ui?.file_path) {
                    filePath = Array.isArray(message.ui.file_path) ? message.ui.file_path[0] : message.ui.file_path;
                    console.log('[SAM3DObjects] Found file_path in message.ui.file_path:', filePath);
                } else if (message?.file_path) {
                    filePath = Array.isArray(message.file_path) ? message.file_path[0] : message.file_path;
                    console.log('[SAM3DObjects] Found file_path in message.file_path:', filePath);
                } else if (this.widgets && this.widgets.length > 0) {
                    // Try to get from widget value
                    const filePathWidget = this.widgets.find(w => w.name === 'file_path');
                    if (filePathWidget) {
                        filePath = filePathWidget.value;
                        console.log('[SAM3DObjects] Found file_path in widget value:', filePath);
                    }
                }
                
                // Use the helper function if we found a file path
                if (filePath && filePath.trim() !== '') {
                    console.log('[SAM3DObjects] Loading point cloud from onExecuted, file path:', filePath);
                    if (this.loadPointCloudFromPath) {
                        this.loadPointCloudFromPath(filePath);
                    } else {
                        console.warn('[SAM3DObjects] loadPointCloudFromPath function not available yet');
                    }
                } else {
                    console.warn('[SAM3DObjects] No file path found in message or widgets');
                }
            };
        }
    }
});

console.log('[SAM3DObjects] VTK Point Cloud Preview extension loaded');

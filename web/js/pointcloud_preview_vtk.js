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
                const iframe = document.createElement("iframe");
                iframe.src = "/extensions/ComfyUI-SAM3DObjects/viewer_vtk.html";
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
                    setValue(v) { }
                });

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
                
                if (message?.file_path && this._vtkIframe) {
                    console.log('[SAM3DObjects] Loading point cloud in VTK viewer from:', message.file_path);

                    // Handle file_path as array
                    const filePath = Array.isArray(message.file_path) ? message.file_path[0] : message.file_path;

                    if (!filePath) return;

                    // Parse file path to extract filename and subfolder
                    // Handle paths like: "output/inference_9/gaussian.ply" or "/full/path/output/inference_9/gaussian.ply"
                    const pathParts = filePath.replace(/^.*\/(output|input)\//, '').split('/');
                    const filename = pathParts.pop(); // Last part is filename
                    const subfolder = pathParts.join('/'); // Rest is subfolder (e.g., "inference_9")

                    // Construct URL to view the file with proper subfolder
                    const url = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=${encodeURIComponent(subfolder)}`;

                    console.log('[SAM3DObjects] File path:', filePath);
                    console.log('[SAM3DObjects] Constructed URL:', url);
                    
                    // Send message to iframe once it's loaded
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

                    // If iframe is already loaded, send immediately
                    if (this._vtkIframe.contentWindow) {
                        console.log('[SAM3DObjects] iframe contentWindow available, sending immediately');
                        sendMessage();
                    } else {
                        console.log('[SAM3DObjects] iframe contentWindow not available yet, waiting for load event');
                    }

                    // Also send on load in case it wasn't ready
                    this._vtkIframe.addEventListener('load', () => {
                        console.log('[SAM3DObjects] iframe load event fired, sending message');
                        sendMessage();
                    }, { once: true });
                }
            };
        }
    }
});

console.log('[SAM3DObjects] VTK Point Cloud Preview extension loaded');

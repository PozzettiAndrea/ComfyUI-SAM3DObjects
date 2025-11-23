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
                container.appendChild(iframe);

                // Store iframe reference
                this._vtkIframe = iframe;

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

                    // Construct URL to view the file
                    // Use output type to access files in output directory
                    // If absolute path is provided, we might need a different approach or ensure it's in output/temp
                    const filename = filePath.split(/[/\]/).pop();
                    const url = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=${encodeURIComponent(filename.split('_')[1]?.split('/')[0] || '')}`;
                    
                    // Actually, the previous logic was:
                    // `/view?filename=${encodeURIComponent(filePath.split('/').pop())}&type=output&subfolder=`
                    // This assumes the file is in the root output directory or handled by ComfyUI's file serving.
                    // Given our SAM3D nodes output to subfolders like 'inference_1', we might need to handle subfolders.
                    // However, `view` endpoint usually requires explicit subfolder param if it's in a subfolder.
                    
                    // Let's try to parse the subfolder from the path if possible, or just pass the filename if it's in root.
                    // But since we are passing an absolute path from the python node, ComfyUI's /view endpoint might not work 
                    // if it's not strictly inside the output directory structure it expects.
                    
                    // Better approach: usage of /view requires filename, type, subfolder.
                    // If our python node returns the absolute path, we should try to serve it.
                    // But the iframe needs a URL. 
                    
                    // Let's assume the standard ComfyUI /view endpoint works for files in output folder.
                    // Our worker saves to `ComfyUI/output/inference_X/mesh.glb` etc.
                    // So subfolder is `inference_X`.
                    
                    // We need to parse subfolder from the absolute path.
                    let subfolder = "";
                    const parts = filePath.split(/[/\]/);
                    const filenameOnly = parts.pop();
                    const parentDir = parts.pop();
                    
                    if (parentDir && parentDir.startsWith("inference_")) {
                        subfolder = parentDir;
                    }
                    
                    const finalUrl = `/view?filename=${encodeURIComponent(filenameOnly)}&type=output&subfolder=${encodeURIComponent(subfolder)}`;

                    console.log('[SAM3DObjects] Constructed URL:', finalUrl);
                    
                    // Send message to iframe once it's loaded
                    const sendMessage = () => {
                        if (this._vtkIframe && this._vtkIframe.contentWindow) {
                            this._vtkIframe.contentWindow.postMessage({
                                type: 'loadPointCloud',
                                url: finalUrl
                            }, '*');
                        }
                    };

                    // If iframe is already loaded, send immediately
                    if (this._vtkIframe.contentWindow) {
                        sendMessage();
                    }

                    // Also send on load in case it wasn't ready
                    this._vtkIframe.addEventListener('load', sendMessage, { once: true });
                }
            };
        }
    }
});

console.log('[SAM3DObjects] VTK Point Cloud Preview extension loaded');

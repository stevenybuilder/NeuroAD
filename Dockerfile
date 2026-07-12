# Static deploy of the REAL-DATA NeuroAD demo for Cloud Run.
# Root "/" is the Claude Science entry (claude_science.html); "Open in NeuroAD"
# opens /neuroad.html (the product). Front-end assets only — NO Python backend,
# NO gated data/weights/secrets. Live /api/* (dynamic Investigate, SFG QC) is not
# served here; the real-data frozen view renders from committed demo_data.json +
# public MNI template + demo ROI masks.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY app/claude_science.html /usr/share/nginx/html/index.html
COPY app/neuroad.html /usr/share/nginx/html/neuroad.html
COPY app/demo_data.json /usr/share/nginx/html/demo_data.json
COPY app/vendor/niivue.umd.js /usr/share/nginx/html/vendor/niivue.umd.js
COPY app/scans/ /usr/share/nginx/html/scans/

EXPOSE 8080

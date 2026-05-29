Sigurdur Haukur Birgisson — ML pipeline & backend
  - ML pipeline foundation: triplet mining, online triplet loss, k-fold
  training
  - Data pipeline: preprocessing filters (Butterworth, Kalman, FFT),
  windowing, standardization
  - Model architectures: LSTM and Transformer implementations
  - Backend/frontend integration: FastAPI routes, web UI, model
  inference
  - DevOps: Docker containerization, Hugging Face integration

Peter Meeus — ML training & evaluation
  - K-fold cross-validation implementation
  - Evaluation metrics: FAR, FRR, EER computation for unseen
  participants
  - Model parameter tuning and finetuning workflow
  - CosFace loss integration (in collaboration with tijje)
  - Training visualization and loss/embedding analysis
  - Dataset handling fixes and preprocessing optimizations
  - DevOps: pre-commit hooks, code formatting, dependency management

Efe — Web API & frontend
  - Initial FastAPI endpoint design and setup
  - Template system and HTML frontend pages
  - API webpage and user interface development
  - Feature separation: model pages, user enrollment, classification
  logic
  - API testing framework
  - Filter experimentation and visualization

Tijje — CosFace loss implementation
  - Custom linear layer for CosFace loss
  - CosFace loss function implementation and integration
  - Alternative loss function support alongside triplet loss
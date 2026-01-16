# Model Config Builder / IR Builder

This directory contains tools and utilities for automatically building model configurations.

## Goals

**Autonomous Model Configuration:**
- Automatically infer model architecture from HuggingFace models
- Extract layer specifications, dimensions, and parameters
- Generate Mantile-compatible JSON configuration files
- Support multiple model families (LLaMA, Mistral, GPT, etc.)

## Current State

_Planning phase - structure and implementation to be defined_

## Proposed Components

### 1. Model Inspection
- Load models from HuggingFace
- Extract architecture details
- Identify layer types and configurations

### 2. Layer Mapping
- Map HF layer types to Mantile layer classes
- Extract relevant parameters (hidden_size, num_heads, etc.)
- Handle architecture-specific variations

### 3. Config Generation
- Generate JSON configuration files
- Validate against Mantile schema
- Support custom overrides and annotations

### 4. Validation
- Test generated configs against actual models
- Verify parameter counts and shapes
- Compare with manual configurations

## Usage

_To be defined_

## Development Plan

See PLAN.md for detailed implementation roadmap.

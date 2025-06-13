# Design Document: Data Discovery and Probabilistic Relationship Analysis

## System Overview

This system implements a probabilistic relationship discovery framework that analyzes text data to identify and quantify relationships between entities and their attributes. The system uses Markov chains and probabilistic graphs to model these relationships, providing both analytical insights and visual representations.

## Architecture

### Core Components

1. **ProbabilisticGraph Class**
   - Manages the relationship graph structure
   - Handles probability calculations and updates
   - Implements visualization capabilities

2. **Markov Chain Implementation**
   - Models state transitions between entities and attributes
   - Computes conditional probabilities
   - Supports bidirectional relationship analysis

3. **NLP Integration**
   - Uses GPT-4 for entity-attribute extraction
   - Processes natural language text into structured relationships
   - Handles schema-based relationship validation

### Data Flow

1. **Input Processing**
   ```
   Text Data → GPT-4 Processing → Entity-Attribute Pairs → Probability Graph
   ```

2. **Probability Computation**
   ```
   Raw Counts → Conditional Probabilities → Bidirectional Relationships
   ```

3. **Output Generation**
   ```
   Probability Graph → JSON Export + Visualization
   ```

## Technical Implementation

### Probability Graph Structure

```python
class ProbabilisticGraph:
    def __init__(self):
        self.graph = defaultdict(dict)        # Final probability graph
        self._counts = defaultdict(int)       # Raw co-occurrence counts
        self._node_count = defaultdict(int)   # Marginal counts
```

### Key Algorithms

1. **Relationship Extraction**
   - Uses GPT-4 to identify entity-attribute pairs
   - Validates against provided schema
   - Computes initial probability estimates

2. **Probability Computation**
   - Calculates P(target|source) and P(source|target)
   - Applies threshold filtering
   - Normalizes probabilities

3. **Graph Visualization**
   - Uses NetworkX for graph representation
   - Implements spring layout for node positioning
   - Generates interactive visualizations

## Data Structures

### Schema Format
```json
{
    "entities": {
        "entity_name": {
            "attributes": ["attr1", "attr2", ...],
            "relationships": ["rel1", "rel2", ...]
        }
    }
}
```

### Probability Graph Format
```json
{
    "source_entity": {
        "target_entity": probability,
        ...
    }
}
```

## Performance Considerations

1. **Memory Management**
   - Uses defaultdict for efficient storage
   - Implements incremental updates
   - Optimizes graph structure for large datasets

2. **Processing Optimization**
   - Batches document processing
   - Implements threshold-based filtering
   - Uses efficient probability calculations

## Security Considerations

1. **API Key Management**
   - Secure storage of OpenAI API key
   - Environment variable-based configuration
   - No hardcoded credentials

2. **Data Privacy**
   - Local processing of sensitive data
   - No data transmission beyond API calls
   - Secure output file handling

## Future Enhancements

1. **Planned Features**
   - Real-time relationship updates
   - Advanced visualization options
   - Custom probability calculation methods

2. **Scalability Improvements**
   - Distributed processing support
   - Enhanced memory optimization
   - Parallel processing capabilities

## Testing Strategy

1. **Unit Tests**
   - Probability calculations
   - Graph operations
   - Data structure integrity

2. **Integration Tests**
   - End-to-end processing
   - API integration
   - File I/O operations

## Error Handling

1. **Input Validation**
   - Schema validation
   - Data format checking
   - API response verification

2. **Recovery Mechanisms**
   - Graceful error handling
   - Data consistency checks
   - Backup and restore capabilities 
"""PAI-OS Pilot Instrumentation (Phase 7.2.5).

Minimal, low-friction observation and feedback mechanism for
real-user continuous usage of PAI-OS.

Components:
- CopilotUsageEvent: Telemetry for each interaction
- Feedback system: GOOD/PARTIAL/BAD + reason codes
- Context usefulness tracking
- Personalization gain measurement
- Learning usefulness tracking
- Research usefulness tracking
- Router decision quality tracking
- Memory feedback tracking
- Daily review feedback tracking
- Recommendation utility tracking
- Privacy sanitizer
- Weekly/daily analysis engine
- REAL_USER / DEMO / EVAL separation
"""
from __future__ import annotations

from .usage_event import (
    CopilotUsageEvent,
    UsageEventStore,
    TaskType,
    UsageMode,
    RetrievalMode,
    DataSource,
)
from .feedback import (
    UserFeedback,
    FeedbackStore,
    FeedbackRating,
    FeedbackReason,
    ContextUsefulnessTag,
    MemoryFeedbackTag,
    DailyReviewUsefulness,
    ManualCorrection,
)
from .personalization import (
    PersonalizationGainTracker,
    PersonalizationScore,
)
from .context_usefulness import (
    ContextUsefulnessTracker,
    ContextFeedbackRecord,
)
from .learning_usefulness import (
    LearningUsefulnessTracker,
    LearningFeedbackRecord,
)
from .research_usefulness import (
    ResearchUsefulnessTracker,
    ResearchFeedbackRecord,
)
from .router_usefulness import (
    RouterUsefulnessTracker,
    RouterDecisionRecord,
)
from .memory_feedback import (
    MemoryFeedbackTracker,
    MemoryFeedbackRecord,
)
from .daily_review_feedback import (
    DailyReviewFeedbackTracker,
    DailyReviewFeedbackRecord,
)
from .recommendation_utility import (
    RecommendationUtilityTracker,
    RecommendationRecord,
)
from .privacy import (
    PrivacySanitizer,
    strip_sensitive_data,
    strip_sensitive_dict,
)
from .analysis import (
    PilotAnalysisEngine,
)
from .store import (
    PilotDataStore,
)

__all__ = [
    # Events
    "CopilotUsageEvent",
    "UsageEventStore",
    "TaskType",
    "UsageMode",
    "RetrievalMode",
    "DataSource",
    # Feedback
    "UserFeedback",
    "FeedbackStore",
    "FeedbackRating",
    "FeedbackReason",
    "ContextUsefulnessTag",
    "MemoryFeedbackTag",
    "DailyReviewUsefulness",
    "ManualCorrection",
    # Context
    "ContextUsefulnessTracker",
    "ContextFeedbackRecord",
    # Personalization
    "PersonalizationGainTracker",
    "PersonalizationScore",
    # Learning
    "LearningUsefulnessTracker",
    "LearningFeedbackRecord",
    # Research
    "ResearchUsefulnessTracker",
    "ResearchFeedbackRecord",
    # Router
    "RouterUsefulnessTracker",
    "RouterDecisionRecord",
    # Memory
    "MemoryFeedbackTracker",
    "MemoryFeedbackRecord",
    # Daily Review
    "DailyReviewFeedbackTracker",
    "DailyReviewFeedbackRecord",
    # Recommendations
    "RecommendationUtilityTracker",
    "RecommendationRecord",
    # Privacy
    "PrivacySanitizer",
    "strip_sensitive_data",
    "strip_sensitive_dict",
    # Analysis
    "PilotAnalysisEngine",
    # Store
    "PilotDataStore",
]

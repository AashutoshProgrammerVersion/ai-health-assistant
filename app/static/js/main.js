// JavaScript for Health Assistant App

document.addEventListener('DOMContentLoaded', function() {
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            if (alert) {
                alert.style.transition = 'opacity 0.5s';
                alert.style.opacity = '0';
                setTimeout(function() {
                    alert.remove();
                }, 500);
            }
        }, 5000);
    });

    // Form validation enhancement
    const forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            // Add loading state to submit buttons
            const submitBtn = form.querySelector('input[type="submit"], button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalText = submitBtn.value || submitBtn.textContent;
                submitBtn.value = 'Processing...';
                submitBtn.textContent = 'Processing...';

                // Re-enable after 3 seconds as fallback
                setTimeout(function() {
                    submitBtn.disabled = false;
                    submitBtn.value = originalText;
                    submitBtn.textContent = originalText;
                }, 3000);
            }
        });
    });
});

// Utility functions for health calculations
function calculateHealthScore(sleepHours, waterIntake, activityLevel, mood) {
    let totalScore = 0;
    let factorsCount = 0;

    // Sleep scoring (optimal: 7-9 hours)
    if (sleepHours !== null && sleepHours !== undefined) {
        let sleepScore;
        if (sleepHours >= 7 && sleepHours <= 9) {
            sleepScore = 10;
        } else if (sleepHours >= 6 && sleepHours < 7 || sleepHours > 9 && sleepHours <= 10) {
            sleepScore = 8;
        } else if (sleepHours >= 5 && sleepHours < 6 || sleepHours > 10 && sleepHours <= 11) {
            sleepScore = 6;
        } else {
            sleepScore = 4;
        }
        totalScore += sleepScore;
        factorsCount++;
    }

    // Water scoring (optimal: 8+ glasses)
    if (waterIntake !== null && waterIntake !== undefined) {
        const waterScore = Math.min(10, Math.max(1, waterIntake));
        totalScore += waterScore;
        factorsCount++;
    }

    // Activity and mood are already 1-10 scales
    if (activityLevel !== null && activityLevel !== undefined) {
        totalScore += activityLevel;
        factorsCount++;
    }

    if (mood !== null && mood !== undefined) {
        totalScore += mood;
        factorsCount++;
    }

    return factorsCount > 0 ? Math.round((totalScore / factorsCount) * 10) / 10 : 0;
}

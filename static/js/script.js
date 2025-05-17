document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('recommendation-form');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            const submitButton = form.querySelector('button[type="submit"]');
            const originalText = submitButton.innerHTML;
            
            // Show loading state
            submitButton.disabled = true;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
            // In a real app, you might want to add error handling here
        });
    }
    
    // Example of how you could add API functionality
    // This would be used if you want to make AJAX calls instead of form submissions
    async function getRecommendations(problemText) {
        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ problem: problemText })
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            throw error;
        }
    }
    
    // Example usage:
    // getRecommendations("Your problem description here")
    //     .then(data => console.log(data))
    //     .catch(error => console.error(error));
});
class CarouselLeaderboard {
  constructor() {
    this.categories = [
      {
        id: 'classification',
        title: 'Classification Tasks Leaderboard',
        dataUrl: 'static/json/leaderboard-classification.json'
      },
      {
        id: 'regression', 
        title: 'Regression Tasks Leaderboard',
        dataUrl: 'static/json/leaderboard-regression.json'
      },
      {
        id: 'description',
        title: 'Molecular Description Leaderboard', 
        dataUrl: 'static/json/leaderboard-description.json'
      }
    ];
    this.currentIndex = 0;
    this.data = null;
    this.sortConfig = { key: 'average', direction: 'desc' };
    
    this.init();
  }
  
  init() {
    this.setupEventListeners();
    this.loadCurrentCategory();
  }
  
  setupEventListeners() {
    // Navigation buttons
    document.getElementById('prev-btn').addEventListener('click', () => this.previousCategory());
    document.getElementById('next-btn').addEventListener('click', () => this.nextCategory());
    
    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach((btn, index) => {
      btn.addEventListener('click', () => this.goToCategory(index));
    });
  }
  
  async loadCurrentCategory() {
    const category = this.categories[this.currentIndex];
    
    try {
      this.data = await this.loadData(category.dataUrl);
      // Reset sort to default for the category type
      this.sortConfig = { key: 'average', direction: this.getDefaultSortDirection() };
      this.updateUI();
      this.renderTable();
    } catch (error) {
      console.error('Error loading data:', error);
      document.getElementById('table-title').textContent = 'Error loading data';
    }
  }
  
  async loadData(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error loading data:', error);
      throw error;
    }
  }
  
  updateUI() {
    // Update title
    document.getElementById('table-title').textContent = this.data.title;
    
    // Update active tab
    document.querySelectorAll('.tab-btn').forEach((btn, index) => {
      btn.classList.toggle('active', index === this.currentIndex);
    });
    
    // Update navigation buttons
    document.getElementById('prev-btn').disabled = this.currentIndex === 0;
    document.getElementById('next-btn').disabled = this.currentIndex === this.categories.length - 1;
  }
  
  goToCategory(index) {
    if (index !== this.currentIndex) {
      this.currentIndex = index;
      this.loadCurrentCategory();
    }
  }
  
  previousCategory() {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      this.loadCurrentCategory();
    }
  }
  
  nextCategory() {
    if (this.currentIndex < this.categories.length - 1) {
      this.currentIndex++;
      this.loadCurrentCategory();
    }
  }
  
  handleSort(key) {
    let direction = 'asc';
    if (this.sortConfig.key === key && this.sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    this.sortConfig = { key, direction };
    this.renderTable();
  }
  
  getDefaultSortDirection() {
    // For regression tasks, lower values are better, so default to ascending
    // For classification and description, higher values are better, so default to descending
    return this.categories[this.currentIndex].id === 'regression' ? 'asc' : 'desc';
  }
  
  renderTable() {
    if (!this.data) return;
    
    // Clear existing content
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';
    
    // Create header
    const headerRow = document.createElement('tr');
    this.data.columns.forEach(column => {
      const th = document.createElement('th');
      th.textContent = column.label;
      
      if (column.sortable) {
        th.classList.add('sortable');
        th.addEventListener('click', () => this.handleSort(column.id));
        
        if (this.sortConfig.key === column.id) {
          th.classList.add(this.sortConfig.direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
        }
      }
      
      if (column.emphasized) {
        th.classList.add('average-column');
      }
      
      headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);
    
    // Sort and create rows
    const sortedRows = [...this.data.rows].sort((a, b) => {
      const aVal = a[this.sortConfig.key];
      const bVal = b[this.sortConfig.key];
      
      if (aVal < bVal) return this.sortConfig.direction === 'asc' ? -1 : 1;
      if (aVal > bVal) return this.sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
    
    sortedRows.forEach(row => {
      const tr = document.createElement('tr');
      
      this.data.columns.forEach(column => {
        const td = document.createElement('td');
        td.textContent = row[column.id];
        
        if (column.id === 'method') {
          td.classList.add('method-column');
        }
        
        if (column.id === 'average') {
          td.classList.add('average-column');
        }
        
        tr.appendChild(td);
      });
      
      tableBody.appendChild(tr);
    });
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new CarouselLeaderboard();
});
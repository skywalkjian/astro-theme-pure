import blogroll from '../../public/links.json'

interface LinkGroup {
  id: string
  title: string
  people: { name: string; link: string; intro: string }[]
}

export const linkGroups: LinkGroup[] = [
  {
    id: 'research-mentors',
    title: 'Research Mentors',
    people: [
      {
        name: 'Weijian Luo',
        link: 'https://pkulwj1994.github.io/',
        intro: 'Supervised my research on image generation at RedNote Dots.'
      },
      {
        name: 'Tianyang Hu',
        link: 'https://hu-tianyang.github.io/',
        intro: 'Supervised my work on diffusion language models at CUHK-Shenzhen.'
      }
    ]
  },
  {
    id: 'collaborators',
    title: 'Collaborators',
    people: [
      {
        name: 'Guang Cheng',
        link: 'https://people.stat.ucla.edu/guangcheng/2.html',
        intro: 'Professor of Statistics and Data Science at UCLA.'
      }
    ]
  },
  {
    id: 'friends',
    title: 'Friends',
    people: blogroll.friends.find((group) => group.id_name === 'cf-links')?.link_list ?? []
  }
]

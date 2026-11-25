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
        name: 'Tianyang Hu',
        link: 'https://hu-tianyang.github.io/',
        intro: 'Supervised my work on diffusion language models at CUHK-Shenzhen.'
      },
      {
        name: 'Weijian Luo',
        link: 'https://pkulwj1994.github.io/',
        intro: 'Supervised my research on image generation at RedNote Dots.'
      }
    ]
  },
  {
    id: 'collaborators',
    title: 'Collaborators',
    people: []
  },
  {
    id: 'friends',
    title: 'Friends',
    people: blogroll.friends.find((group) => group.id_name === 'cf-links')?.link_list ?? []
  }
]
